//===- CIMCostReport.cpp - Pass 8: emit JSON cost report -------*- C++ -*-===//

#include "PassDetail.h"
#include "cim/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
struct CIMCostReportPass : public CIMCostReportBase<CIMCostReportPass> {
  void runOnOperation() override {
    // TODO(spec Sec.6, Pass 8): walk the final IR, sum cim.program's
    // cost_ns/cost_pj and cim.mvm's target-file-derived cost, emit JSON
    // (spec Sec. 17 worked example format: install cost vs. amortized
    // per-inference cost). This is not an optimization pass — it produces
    // the project's publishable numbers.
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMCostReportPass() {
  return std::make_unique<CIMCostReportPass>();
}
