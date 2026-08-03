//===- CIMInsertTransfers.cpp - Pass 5: explicit data movement -*- C++ -*-===//

#include "PassDetail.h"
#include "cim/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
struct CIMInsertTransfersPass
    : public CIMInsertTransfersBase<CIMInsertTransfersPass> {
  void runOnOperation() override {
    // TODO(spec Sec.6, Pass 5): walk the IR; wherever an op requires an
    // operand in #cim.space<near> but it lives in #cim.space<host> (or any
    // other space mismatch), insert cim.copy. Then run a small dataflow
    // analysis to hoist redundant copies out of loops.
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMInsertTransfersPass() {
  return std::make_unique<CIMInsertTransfersPass>();
}
