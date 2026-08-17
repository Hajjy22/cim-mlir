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

  module.walk([&](mlir::cim::RequantizeOp op) {
    FailureOr<std::pair<int64_t, int>> counted = weightedCount(op);
    if (failed(counted)) {
      ++counts.unknownRequantizeSites;
      return;
    }
    counts.requantizes += static_cast<uint64_t>(counted->first);
  });

  module.walk([&](mlir::cim::ReducePartialOp op) {
    FailureOr<std::pair<int64_t, int>> counted = weightedCount(op);
    if (failed(counted)) {
      ++counts.unknownReduceSites;
      return;
    }
    // N operands lower to N-1 chained cimrt_reduce_add calls
    // (lowerReducePartial, lib/Transforms/CIMLowerToTarget.cpp) -- weight
    // the site by that many calls, not by one, or a reduce_partial summing
    // four partials would be charged the same as one summing two.
    const size_t numPartials = op.getPartials().size();
    const uint64_t addsPerFiring =
        numPartials > 0 ? static_cast<uint64_t>(numPartials - 1) : 0;
    counts.reduceAdds += static_cast<uint64_t>(counted->first) * addsPerFiring;
  });

  module.walk([&](mlir::cim::ReduceMaxOp op) {
    FailureOr<std::pair<int64_t, int>> counted = weightedCount(op);
    if (failed(counted)) {
      ++counts.unknownReduceSites;
      return;
    }
    // Identical N-1 weighting to reduce_partial above, for the identical
    // reason: a Kh*Kw pooling window is ONE op site that issues Kh*Kw-1
    // cimrt_reduce_max calls, and the runtime counts calls. Charged into
    // its own counter, not reduceAdds, because the two are billed against
    // different target-file cost entries.
    const size_t numInputs = op.getInputs().size();
    const uint64_t maxesPerFiring =
        numInputs > 0 ? static_cast<uint64_t>(numInputs - 1) : 0;
    counts.reduceMaxes += static_cast<uint64_t>(counted->first) * maxesPerFiring;
  });

  module.walk([&](mlir::cim::CopyOp op) {
    FailureOr<std::pair<int64_t, int>> counted = weightedCount(op);
    if (failed(counted)) {
      ++counts.unknownCopySites;
      return;
    }

    auto srcType = llvm::dyn_cast<MemRefType>(op.getSource().getType());
    auto destType = llvm::dyn_cast<MemRefType>(op.getDest().getType());
    if (!srcType || !destType) {
      ++counts.unknownCopySites;
      return;
    }

    // Mirrors lowerCimCopy's own host/device test (CIMLowerToTarget.cpp)
    // exactly, because the point of this count is to charge precisely the
    // transfers that lowering routes through cimrt -- a different rule
    // here would produce a static number that describes a different
    // program than the one that runs.
    auto spaceOf = [](MemRefType t) {
      return llvm::dyn_cast_or_null<mlir::cim::SpaceAttr>(t.getMemorySpace());
    };
    auto isDevice = [&](MemRefType t) {
      auto space = spaceOf(t);
      return space && space.getKind() != mlir::cim::SpaceKind::host;
    };

    if (!isDevice(srcType) && !isDevice(destType)) {
      // Host to host: lowerCimCopy short-circuits this to a plain
      // memref.copy that never reaches cimrt, so the runtime charges it
      // nothing. Counted separately rather than folded into zero -- see
      // IRCostCounts::transferBytes.
      counts.hostToHostCopies += static_cast<uint64_t>(counted->first);
      return;
    }

    // lowerCimCopy sizes a device->host transfer by its SOURCE and every
    // other case by its destination; match that, or a copy between
    // differently-sized views would be charged the wrong way round.
    const bool deviceToHost = isDevice(srcType) && !isDevice(destType);
    MemRefType sizingType = deviceToHost ? srcType : destType;
    if (!sizingType.hasStaticShape()) {
      ++counts.unknownCopySites;
      return;
    }
    auto elemType = llvm::dyn_cast<IntegerType>(sizingType.getElementType());
    if (!elemType || elemType.getWidth() % 8 != 0) {
      ++counts.unknownCopySites;
      return;
    }
    const uint64_t bytesPerFiring =
        static_cast<uint64_t>(sizingType.getNumElements()) *
        (elemType.getWidth() / 8);
    counts.transferBytes +=
        static_cast<uint64_t>(counted->first) * bytesPerFiring;
  });

  return counts;
}
