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
// 3. CROSS-ITERATION REUSE IS A SEPARATE STEP: LOOP HOISTING. Step 1-2 above
//    find reuse WITHIN one textual execution of a block. When that block is
//    an scf.for's body, the block still executes once *textually* but many
//    times *at runtime* -- reuse ACROSS those runtime executions is a
//    different question, answered after every block has already been placed:
//
//    A surviving cim.program op (one Step-3 kept, i.e. not eliminated as an
//    intra-iteration reuse) is a hoisting candidate iff it is the ONLY
//    cim.program in its block writing to its physical tile. If two
//    surviving programs share a tile id, that tile's content already
//    changes WITHIN one iteration, so it cannot be assumed stable ACROSS
//    iterations either -- the same argument as intra-block reuse, one level
//    up. This is checked directly off `result.actions` (see
//    hoistCandidates), not re-derived.
//
//    A candidate is only actually hoisted out of an enclosing scf.for when
//    that loop's trip count is a compile-time-provable positive number
//    (loopHasProvablyPositiveTripCount) -- a loop that might run zero times
//    would make the hoisted cim.program fire even when the body never does,
//    which changes nothing about any OUTPUT VALUE (an unused tile write is
//    inert) but does spend energy/latency the original IR never spent, so it
//    is refused rather than risked. Nested loops are left alone entirely for
//    v0.1: only the innermost-relevant single level is reasoned about.
//
//    The mechanics are mlir::moveLoopInvariantCode (a stock MLIR utility),
//    supplied with cim-specific predicates: cim.device_open, cim.tile_alloc,
//    memref.get_global/subview/cast are always safe to hoist when their
//    operands are loop-invariant (device_open and tile_alloc have no
//    declared memory effects that forbid it and are idempotent by name/id;
//    the memref ops are pure views), and cim.program is safe to hoist only
//    when it is in `hoistCandidates`. Everything else -- cim.mvm,
//    cim.reduce_partial, cim.copy, memref.copy -- is never in the allowed
//    set, so activation-side computation (which genuinely differs every
//    iteration) never moves; the operand-invariance check would refuse it
//    anyway, since it reads a per-iteration value, but the op-kind
//    allowlist is the belt to that suspenders.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Placement/Placement.h"
#include "cim/Target/TargetSpec.h"
#include "cim/Transforms/LoopAnalysis.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

// loopHasProvablyPositiveTripCount now lives in cim/Transforms/LoopAnalysis.h,
// shared with cim-cost-report (spec Sec. 6, Pass 8), which needs the exact
// trip count (getConstantTripCount) rather than just this boolean to weight
// ops inside a loop.
using ::mlir::cim::loopHasProvablyPositiveTripCount;

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

    // Populated by placeBlock with the cim.program ops proven safe for
    // cross-iteration reuse -- see decision 3 in the file header. Collected
    // across every block before any hoisting happens, so hoisting never
    // interleaves with (and cannot disturb) the per-block rewrite above.
    llvm::DenseSet<Operation *> hoistCandidates;

    for (auto &entry : byBlock)
      if (failed(placeBlock(entry.second, spec, hoistCandidates)))
        return signalPassFailure();

    hoistFromLoops(module, hoistCandidates);
  }

  /// Move every hoisting candidate that survived placeBlock out of its
  /// enclosing scf.for, when that loop provably runs. One level only: a
  /// loop nested inside another scf.for is left untouched (its candidates,
  /// if any, stay exactly where cim-partition put them) rather than reasoned
  /// about multi-level, which is future work, not this pass's v0.1 scope.
  void hoistFromLoops(ModuleOp module,
                      const llvm::DenseSet<Operation *> &hoistCandidates) {
    module.walk([&](scf::ForOp forOp) {
      const bool nested = forOp->getParentOfType<scf::ForOp>() != nullptr;
      const bool tripCountKnown = loopHasProvablyPositiveTripCount(forOp);

      // Only warn when skipping cost this loop something real: most loops
      // in the world have no cim.program in them at all, and flagging every
      // one of those would be pure noise.
      bool hasCandidate = false;
      if (nested || !tripCountKnown)
        forOp.getRegion().walk([&](ProgramOp op) {
          if (hoistCandidates.contains(op.getOperation()))
            hasCandidate = true;
        });

      if (nested) {
        if (hasCandidate)
          forOp.emitRemark(
              "cim-placement: not hoisting cim.program out of a loop nested "
              "inside another scf.for; only one level is handled in v0.1");
        return;
      }
      if (!tripCountKnown) {
        if (hasCandidate)
          forOp.emitRemark(
              "cim-placement: not hoisting cim.program out of this scf.for; "
              "its trip count is not provably positive, and a loop that "
              "might run zero times must not be assumed to run at all");
        return;
      }

      Region &region = forOp.getRegion();
      mlir::moveLoopInvariantCode(
          {&region},
          [](Value v, Region *r) {
            return v.getParentRegion()->isProperAncestor(r);
          },
          [&](Operation *op, Region *) {
            // device_open and tile_alloc declare no memory effects that
            // forbid motion, and are idempotent (device_open by name --
            // see Interpreter::deviceFor's cache -- tile_alloc by id, a
            // pure lookup with no device-touching side effect). The memref
            // ops are pure views. cim.program is the one case that
            // genuinely needs proof, supplied by placeBlock's per-tile
            // count, not by anything generic here.
            if (isa<DeviceOpenOp, TileAllocOp, memref::GetGlobalOp,
                    memref::SubViewOp, memref::CastOp>(op))
              return true;
            if (auto program = dyn_cast<ProgramOp>(op))
              return hoistCandidates.contains(program.getOperation());
            return false;
          },
          [&](Operation *op, Region *) { op->moveBefore(forOp); });
    });
  }

  LogicalResult placeBlock(ArrayRef<ProgramOp> programs,
                           const ::cim::TargetSpec &spec,
                           llvm::DenseSet<Operation *> &hoistCandidates) {
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

    // How many surviving cim.program ops target each physical tile within
    // THIS block. A tile programmed more than once here has content that
    // already changes mid-iteration, so nothing programmed into it can be
    // assumed stable across iterations either -- see decision 3 in the file
    // header. Computed once, off the schedule, rather than re-derived from
    // the rewritten IR below.
    llvm::DenseMap<::cim::TileId, unsigned> programCountPerTile;
    for (const ::cim::PlacementAction &action : result.actions)
      if (action.kind != ::cim::ActionKind::Reuse)
        ++programCountPerTile[action.tile];

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

      if (programCountPerTile.lookup(action.tile) == 1)
        hoistCandidates.insert(op.getOperation());
    }

    return success();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMPlacementPass() {
  return std::make_unique<CIMPlacementPass>();
}
