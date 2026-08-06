//===- CIMPlacement.cpp - Pass 3: Belady tile placement --------*- C++ -*-===//
//
// THE FLAGSHIP PASS (spec Sec. 3.3, 6). Everything else in this pipeline is
// plumbing; this is the pass that justifies the project.
//
// The scheduling algorithm lives in lib/Placement (namespace `cim`, no MLIR
// dependency) and is proven optimal against exhaustive search in
// test/unit/placement_property_test.cpp. This file is the adapter: recover
// the execution order of weight sub-matrices from the IR, hand it to the
// engine, and rewrite cim.program ops from the schedule it returns.
//
// Two decisions carry the whole pass:
//
// 1. WEIGHT IDENTITY IS NOT SSA IDENTITY. cim-partition emits a fresh
//    memref.subview per block per matmul, so two matmuls reading the same
//    weight global produce different SSA values for the same physical
//    weights. Keying on the Value would make every weight distinct, the use
//    sequence would have no repeats, and Belady would have nothing to win --
//    the pass would run, validate, rewrite nothing, and look like it worked.
//    Identity is therefore (root allocation, byte offset, shape), recovered
//    by walking the subview/cast chain to its root and reading the absolute
//    offset off the strided layout MLIR already composed.
//
// 2. PLACEMENT IS PER BLOCK, NOT PER MODULE. Reuse means rewiring a later
//    cim.mvm to consume a resident SSA value produced by an earlier
//    cim.program. That is only valid if the earlier value dominates the
//    later use. Solving one block at a time makes program order the
//    dominance order and removes the question entirely; a resident cannot
//    cross a function boundary anyway.
//
// OPTIMALITY IS INHERITED, NOT RE-ARGUED HERE. This pass emits exactly
// `result.programs` cim.program ops, and `result.programs` is proven minimal
// against exhaustive search by test/unit/placement_property_test.cpp. So the
// only open question is whether the use sequence recovered above is the right
// one -- which is what decision 1 is about, and what the multi-matmul cases in
// test/mlir/pipeline_e2e_test.cpp check.
//
// SCOPE: straight-line code. Reuse is found across matmuls within a block, not
// across iterations of a loop. The "a model that fits reprograms nothing per
// inference" claim therefore still lives only in the cim-bench simulator;
// hoisting cim.program out of an scf.for is separate work (docs/roadmap.md).
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Placement/Placement.h"
#include "cim/Target/TargetSpec.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

#define DEBUG_TYPE "cim-placement"

using namespace mlir;
using namespace mlir::cim;

namespace {

/// Walk a memref back through views to the allocation it ultimately refers
/// to. memref.subview and memref.cast are the only view-forming ops
/// cim-partition emits; anything else is treated as a root, which is the
/// conservative answer (it can only split weights that were really the
/// same, costing a missed reuse, never merging weights that differ).
Value rootAllocationOf(Value memref) {
  Value current = memref;
  while (Operation *def = current.getDefiningOp()) {
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      current = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      current = cast.getSource();
      continue;
    }
    break;
  }
  return current;
}

/// A canonical string identifying the physical weights a memref denotes.
/// Two cim.program ops with equal keys program the same bytes into a tile,
/// so the second is a candidate for elimination.
///
/// A string rather than a struct with a DenseMapInfo: this runs once per
/// cim.program and the clarity is worth more than the allocation.
FailureOr<std::string> weightIdentity(Operation *op, Value weights) {
  auto type = dyn_cast<MemRefType>(weights.getType());
  if (!type || !type.hasStaticShape()) {
    op->emitError("cim-placement needs a statically shaped weight memref to "
                  "identify what a tile holds");
    return failure();
  }

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  // Free function in MLIR 18; becomes MemRefType::getStridesAndOffset in
  // MLIR >= 20. The offset is absolute with respect to the root allocation,
  // because MLIR composed it when it inferred each subview's result type.
  if (failed(getStridesAndOffset(type, strides, offset)) ||
      ShapedType::isDynamic(offset)) {
    op->emitError("cim-placement cannot identify a weight sub-matrix with a "
                  "dynamic offset or stride; it would have to guess whether "
                  "two tiles hold the same weights");
    return failure();
  }

  const Value root = rootAllocationOf(weights);

  std::string key;
  llvm::raw_string_ostream os(key);
  // A global is named, so two functions referring to @w agree. Anything
  // else is identified by the pointer of its defining op, which is stable
  // for the lifetime of this pass and cannot collide across roots.
  if (auto global = root.getDefiningOp<memref::GetGlobalOp>())
    os << "@" << global.getName();
  else if (Operation *def = root.getDefiningOp())
    os << "op:" << static_cast<const void *>(def) << ":"
       << cast<OpResult>(root).getResultNumber();
  else
    os << "arg:" << static_cast<const void *>(root.getAsOpaquePointer());

  os << "+" << offset << "x[";
  llvm::interleave(type.getShape(), os, "x");
  os << "]:" << type.getElementType();
  os.flush();
  return key;
}

/// Erase `value`'s defining op, and its sources, while they are dead views.
/// Rewiring a cim.mvm away from an eliminated cim.program usually orphans
/// the memref.subview that addressed those weights; leaving it behind would
/// make the output IR read as though a transfer still happens.
void eraseDeadViewChain(Value value) {
  while (value && value.use_empty()) {
    Operation *def = value.getDefiningOp();
    if (!def)
      return;
    Value source;
    if (auto subview = dyn_cast<memref::SubViewOp>(def))
      source = subview.getSource();
    else if (auto cast = dyn_cast<memref::CastOp>(def))
      source = cast.getSource();
    else
      return;
    def->erase();
    value = source;
  }
}

struct CIMPlacementPass : public CIMPlacementBase<CIMPlacementPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (targetYAML.empty()) {
      module.emitError("cim-placement requires -target-yaml=<path>; the "
                       "number of physical tiles to place into comes from the "
                       "target description (spec Sec. 7)");
      return signalPassFailure();
    }
    ::cim::TargetSpec spec;
    std::string error;
    if (!::cim::parseTargetSpecFromFile(targetYAML, spec, &error)) {
      module.emitError("cim-placement: ") << error;
      return signalPassFailure();
    }
    // Defence in depth, and deliberately untested: parseTargetSpecFromFile
    // already rejects tiles.count == 0 ("must be greater than zero"), so no
    // target file can reach this. It stays because computePlacement returns
    // an empty result for numTiles == 0, which would look like "nothing to
    // place" rather than an error, and a TargetSpec built programmatically
    // rather than parsed has no such guarantee.
    if (spec.tiles.count == 0) {
      module.emitError("cim-placement: the target declares zero tiles, so "
                       "there is nowhere to place weights");
      return signalPassFailure();
    }

    // Group by block: see the dominance note in the file header. MapVector
    // keeps blocks in walk order so diagnostics come out deterministically.
    llvm::MapVector<Block *, SmallVector<ProgramOp>> byBlock;
    module.walk([&](ProgramOp op) { byBlock[op->getBlock()].push_back(op); });

    for (auto &entry : byBlock)
      if (failed(placeBlock(entry.second, spec)))
        return signalPassFailure();
  }

  LogicalResult placeBlock(ArrayRef<ProgramOp> programs,
                           const ::cim::TargetSpec &spec) {
    // Step 1 -- recover the use sequence. Each cim.program in block order is
    // one use of the weight sub-matrix it programs. Because the model graph
    // is static, this order is fully known at compile time, which is exactly
    // what makes optimal (Belady) eviction implementable rather than merely
    // desirable.
    ::cim::PlacementProblem problem;
    problem.numTiles = spec.tiles.count;
    problem.name = "cim-placement";

    llvm::StringMap<::cim::WeightId> idForKey;
    for (ProgramOp op : programs) {
      FailureOr<std::string> key = weightIdentity(op, op.getWeights());
      if (failed(key))
        return failure();
      const auto it = idForKey.try_emplace(
          *key, static_cast<::cim::WeightId>(idForKey.size())).first;
      problem.useSequence.push_back(it->second);
    }

    if (problem.useSequence.empty())
      return success();

    // Step 2 -- solve, then check the answer before acting on it.
    const ::cim::PlacementResult result =
        ::cim::computePlacement(problem, ::cim::EvictionPolicy::Belady);

    std::string error;
    if (!::cim::validatePlacement(problem, result, &error)) {
      ProgramOp(programs.front())->emitError(
          "cim-placement produced an invalid schedule: ")
          << error;
      return failure();
    }
    if (result.actions.size() != programs.size()) {
      ProgramOp(programs.front())->emitError(
          "cim-placement: the schedule has ")
          << result.actions.size() << " actions for " << programs.size()
          << " cim.program ops";
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "cim-placement: " << result.programs << " programs, "
               << result.reuses << " reuses, " << result.evictions
               << " evictions over " << problem.useSequence.size()
               << " steps (" << result.distinctWeights
               << " distinct weights, " << problem.numTiles << " tiles)\n");

    // Step 3 -- rewrite. Walking forward in block order means the resident
    // value recorded for a tile was always produced by an earlier op, so it
    // dominates every use rewired to it.
    llvm::DenseMap<::cim::TileId, Value> residentForTile;
    llvm::DenseMap<::cim::TileId, ::cim::WeightId> weightInTile;

    for (size_t index = 0; index < programs.size(); ++index) {
      // A local copy, not a reference into the ArrayRef: MLIR op wrappers
      // are value types holding an Operation*, and their accessors are
      // non-const, so a `const ProgramOp &` cannot call any of them.
      ProgramOp op = programs[index];
      const ::cim::PlacementAction &action = result.actions[index];

      auto tileAlloc = op.getTile().getDefiningOp<TileAllocOp>();
      if (!tileAlloc) {
        op->emitError("cim-placement: this cim.program's tile did not come "
                      "from a cim.tile_alloc, so there is no tile id to "
                      "assign");
        return failure();
      }

      if (action.kind == ::cim::ActionKind::Reuse) {
        // The engine says these weights are already in `action.tile`. It
        // has been replayed by validatePlacement, but a rewrite that
        // silently rewires an mvm to the wrong weights is precisely the
        // catastrophic miscompile this dialect exists to prevent, so the
        // claim is checked once more against what we actually recorded.
        auto resident = residentForTile.find(action.tile);
        auto held = weightInTile.find(action.tile);
        if (resident == residentForTile.end() ||
            held == weightInTile.end() || held->second != action.weight) {
          op->emitError("cim-placement: the schedule claims a reuse from tile ")
              << action.tile << ", but that tile does not hold weight "
              << action.weight << " at this point";
          return failure();
        }

        Value weights = op.getWeights();
        op.getResult().replaceAllUsesWith(resident->second);
        op->erase();
        if (tileAlloc->use_empty())
          tileAlloc->erase();
        eraseDeadViewChain(weights);
        continue;
      }

      tileAlloc.setId(static_cast<uint64_t>(action.tile));
      residentForTile[action.tile] = op.getResult();
      weightInTile[action.tile] = action.weight;
    }

    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMPlacementPass() {
  return std::make_unique<CIMPlacementPass>();
}
