//===- CIMSchedule.cpp - Pass 4: order emitted cim ops ---------*- C++ -*-===//

#include "PassDetail.h"
#include "cim/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
struct CIMSchedulePass : public CIMScheduleBase<CIMSchedulePass> {
  void runOnOperation() override {
    // TODO(spec Sec.6, Pass 4): v0.1 keeps source order and inserts
    // cim.barrier conservatively. v0.2: overlap cim.program of tile i+1
    // with cim.mvm on tile i where the target's
    // capabilities.double_buffer_program is true (spec Sec. 7); report the
    // achievable overlap from the target file.
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMSchedulePass() {
  return std::make_unique<CIMSchedulePass>();
}
