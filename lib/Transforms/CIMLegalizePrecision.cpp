//===- CIMLegalizePrecision.cpp - Pass 6: requantize legalization -*- C++ -*-//

#include "PassDetail.h"
#include "cim/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
struct CIMLegalizePrecisionPass
    : public CIMLegalizePrecisionBase<CIMLegalizePrecisionPass> {
  void runOnOperation() override {
    // TODO(spec Sec.6, Pass 6): insert cim.requantize after every
    // cim.reduce_partial. Read precision.output_effective_bits from the
    // target file (spec Sec. 7). If effective_bits < 8, emit a warning with
    // the estimated accuracy impact rather than silently degrading.
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMLegalizePrecisionPass() {
  return std::make_unique<CIMLegalizePrecisionPass>();
}
