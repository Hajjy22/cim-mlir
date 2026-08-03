//===- CIMPlacement.cpp - Pass 3: Belady tile placement --------*- C++ -*-===//
//
// THE FLAGSHIP PASS (spec Sec. 3.3, 6). Everything else in this pipeline is
// plumbing; this is the pass that justifies the project.
//
// The scheduling algorithm itself lives in lib/Placement (namespace `cim`,
// no MLIR dependency) and is unit-tested without a toolchain build. This
// file is only the adapter: walk the IR to recover the execution order of
// weight sub-matrices, hand that to the engine, and rewrite cim.program ops
// from the schedule it returns.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Placement/Placement.h"
#include "cim/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cim-placement"

using namespace mlir;
using namespace mlir::cim;

namespace {

struct CIMPlacementPass : public CIMPlacementBase<CIMPlacementPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Step 1 — recover the use sequence.
    //
    // Each cim.program in the module introduces a weight sub-matrix, and
    // each cim.mvm consumes one. Walking in program order gives the fixed
    // execution order the engine needs; because the model graph is static,
    // that order is fully known here, which is exactly what makes optimal
    // (Belady) eviction implementable rather than merely desirable.
    ::cim::PlacementProblem problem;
    llvm::DenseMap<Value, ::cim::WeightId> weightIds;
    SmallVector<ProgramOp> programOps;

    module.walk([&](ProgramOp op) {
      const auto id = static_cast<::cim::WeightId>(weightIds.size());
      weightIds.try_emplace(op.getWeights(), id);
      programOps.push_back(op);
    });

    module.walk([&](MvmOp op) {
      // TODO: map the !cim.resident operand back to the weight memref it was
      // programmed from. Requires walking the resident value's defining
      // cim.program, which is well-defined thanks to verifier rule 3
      // (spec Sec. 5.4) making residency an SSA value rather than a side
      // effect on a mutable tile.
      (void)op;
    });

    // TODO: read tiles.count from the target file rather than assuming.
    // Blocked on threading a TargetSpec through the pass pipeline, which
    // cim-lower-to-target's `target-yaml` option will also need.
    if (problem.useSequence.empty() || problem.numTiles == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "cim-placement: no placement problem recovered from the IR; "
                    "the pass is a no-op until cim-partition emits cim.program "
                    "ops and the target file is threaded through\n");
      return;
    }

    // Step 2 — solve. This is the part that is already real and tested; see
    // test/unit/placement_test.cpp.
    const ::cim::PlacementResult result =
        ::cim::computePlacement(problem, ::cim::EvictionPolicy::Belady);

    // Never trust a schedule we are about to rewrite the IR from.
    std::string error;
    if (!::cim::validatePlacement(problem, result, &error)) {
      module.emitError("cim-placement produced an invalid schedule: ") << error;
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "cim-placement: " << result.programs << " programs, "
               << result.reuses << " reuses, " << result.evictions
               << " evictions over " << problem.useSequence.size() << " steps\n");

    // Step 3 — rewrite.
    // TODO: erase the cim.program ops the schedule marks as Reuse, assign
    // tile IDs from PlacementAction::tile on those it keeps, and re-thread
    // the !cim.resident SSA values so each cim.mvm consumes the handle for
    // the tile the schedule placed its weights in.
    (void)programOps;
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMPlacementPass() {
  return std::make_unique<CIMPlacementPass>();
}
