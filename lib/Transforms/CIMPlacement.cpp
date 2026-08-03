//===- CIMPlacement.cpp - Pass 3: Belady tile placement --------*- C++ -*-===//
//
// THE FLAGSHIP PASS (spec Sec. 3.3, 6). Everything else in this pipeline is
// plumbing; this is the pass that justifies the project.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
struct CIMPlacementPass : public CIMPlacementBase<CIMPlacementPass> {
  void runOnOperation() override {
    // TODO(spec Sec.3.3/6, Pass 3): given T physical tiles and W weight
    // sub-matrices where W > T, and a static execution order (known at
    // compile time -> perfect future knowledge), assign sub-matrices to
    // tiles to minimize total cim.program cost:
    //
    //   for each weight sub-matrix W_i in execution order:
    //     if W_i already resident in some tile:
    //       reuse it, emit no cim.program                      // the win
    //     else if a free tile exists:
    //       assign, emit cim.program
    //     else:
    //       evict the resident weight whose NEXT USE is furthest in the
    //       future (Belady / MIN eviction)
    //       assign, emit cim.program
    //
    // For a model that fits entirely in tiles, this eliminates ALL
    // reprogramming after the first pass — on non-volatile hardware the
    // weights are programmed once at install time and never again. That
    // is the headline result (spec Sec. 17 worked example).
    //
    // v0.2+ (not this pass): relax the fixed execution order via ILP
    // (OR-Tools) against the greedy baseline here.
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMPlacementPass() {
  return std::make_unique<CIMPlacementPass>();
}
