//===- LoopAnalysis.cpp - Shared scf.for trip-count queries -----*- C++ -*-=//
//
// See the header for why this is shared between cim-placement and
// cim-cost-report.
//
//===----------------------------------------------------------------------===//
#include "cim/Transforms/LoopAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/APInt.h"

using namespace mlir;

FailureOr<int64_t> mlir::cim::getConstantTripCount(scf::ForOp forOp) {
  APInt lower, upper, step;
  if (!matchPattern(forOp.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(forOp.getUpperBound(), m_ConstantInt(&upper)) ||
      !matchPattern(forOp.getStep(), m_ConstantInt(&step)))
    return failure();
  if (!step.sgt(0))
    return failure();
  if (!lower.slt(upper))
    return 0;
  // ceil((upper - lower) / step), matching scf.for's documented semantics
  // (the induction variable runs lower, lower+step, ... while < upper).
  // Widen to avoid the subtraction overflowing the operands' own bit width
  // before the division narrows it back down.
  const APInt diff = (upper - lower).sext(upper.getBitWidth() + 1);
  const APInt wideStep = step.sext(step.getBitWidth() + 1);
  const APInt count = (diff + wideStep - 1).sdiv(wideStep);
  return count.getSExtValue();
}

bool mlir::cim::loopHasProvablyPositiveTripCount(scf::ForOp forOp) {
  FailureOr<int64_t> count = getConstantTripCount(forOp);
  return succeeded(count) && *count > 0;
}
