//===- CIMDetect.cpp - Pass 1: detect CIM candidate matmuls ----*- C++ -*-===//

#include "PassDetail.h"
#include "cim/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
struct CIMDetectPass : public CIMDetectBase<CIMDetectPass> {
  void runOnOperation() override {
    // TODO(spec Sec.6, Pass 1): pattern-match linalg.matmul / batch_matmul
    // (later: conv_2d) where operands are INT8 and one operand is a
    // constant (i.e. a weight). Annotate with {cim.candidate = true}.
    // Do NOT convert to cim ops here — detection and conversion are
    // separate passes so both are independently testable.
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMDetectPass() {
  return std::make_unique<CIMDetectPass>();
}
