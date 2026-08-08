//===- CostReportUtils.cpp - IR-driven op counting for Pass 8 ---*- C++ -*-=//
//
// See the header for why this is split out from the cim-cost-report pass.
//
//===----------------------------------------------------------------------===//
#include "cim/Transforms/CostReportUtils.h"

#include "cim/Dialect/CIMOps.h"
#include "cim/Transforms/LoopAnalysis.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include <utility>

using namespace mlir;

namespace {

/// One op's execution count for the whole run, and how many scf.for levels
/// enclose it. Failure means some enclosing loop's trip count is not a
/// compile-time constant -- the op's true execution count has no static
/// answer, and this pass never guesses one (matching the interpreter's own
/// "never produce a plausible number it cannot stand behind" rule,
/// lib/Interpreter/Interpreter.cpp).
FailureOr<std::pair<int64_t, int>> weightedCount(Operation *op) {
  int64_t weight = 1;
  int depth = 0;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp)
      continue;
    FailureOr<int64_t> trip = mlir::cim::getConstantTripCount(forOp);
    if (failed(trip))
      return failure();
    weight *= *trip;
    ++depth;
  }
  return std::make_pair(weight, depth);
}

} // namespace

mlir::cim::IRCostCounts mlir::cim::countWeightedOps(ModuleOp module) {
  IRCostCounts counts;

  module.walk([&](mlir::cim::ProgramOp op) {
    FailureOr<std::pair<int64_t, int>> counted = weightedCount(op);
    if (failed(counted)) {
      ++counts.unknownProgramSites;
      return;
    }
    const auto [weight, depth] = *counted;
    counts.programs += static_cast<uint64_t>(weight);
    if (depth == 0)
      ++counts.installPrograms;
    else if (depth == 1)
      ++counts.steadyStatePrograms;
    // depth >= 2: counted in `programs` above (its weight is fully known)
    // but not classified into install vs. steady-state -- see the header.
  });

  module.walk([&](mlir::cim::MvmOp op) {
    FailureOr<std::pair<int64_t, int>> counted = weightedCount(op);
    if (failed(counted)) {
      ++counts.unknownMvmSites;
      return;
    }
    counts.mvms += static_cast<uint64_t>(counted->first);
  });

  return counts;
}
