//===- CIMLowerToTarget.cpp - Pass 7: lower to cimrt calls -----*- C++ -*-===//

#include "PassDetail.h"
#include "cim/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
struct CIMLowerToTargetPass
    : public CIMLowerToTargetBase<CIMLowerToTargetPass> {
  void runOnOperation() override {
    // TODO(spec Sec.6, Pass 7): parse `targetYAML` (lib/Target) and convert
    // each cim op to the matching cimrt_* runtime call (cim.device_open ->
    // cimrt_open, cim.program -> cimrt_program, cim.mvm -> cimrt_mvm, etc).
    // For v0.1, lower straight to runtime calls; a target dialect is
    // premature until there are two real targets (spec M4).
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMLowerToTargetPass() {
  return std::make_unique<CIMLowerToTargetPass>();
}
