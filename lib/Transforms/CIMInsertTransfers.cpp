//===- CIMInsertTransfers.cpp - Pass 5: explicit data movement -*- C++ -*-===//
//
// Spec Sec. 6, Pass 5. cim.mvm's activations operand must live in
// #cim.space<near> (spec Sec. 3.4) -- not yet enforced by
// MvmOp::verify() itself, precisely because this is the pass responsible
// for making it true (see the "Rule 2" note in lib/Dialect/CIM/IR/CIMOps.cpp:
// requiring it before this pass exists would reject every hand-written
// test). Wherever an mvm's activation is not already near-space, this pass
// inserts a cim.copy and rewires the mvm to read the copy's result instead.
//
// Followed by the "small dataflow analysis to hoist redundant copies out of
// loops" spec Sec. 6 also calls for: when the activation being copied is
// loop-invariant with respect to the mvm's nearest enclosing scf.for (its
// home region is not the loop body or anything nested inside it), the copy
// is inserted once, immediately before the loop, rather than once per
// mvm per iteration -- and shared by every mvm inside that loop reading the
// exact same invariant source, rather than duplicated per site. v0.1
// handles one level of loop nesting, the same scope cim-placement's own
// hoisting documents (never out of a loop nested inside another).
//
// On today's real pipeline this pass has nothing to do: cim-partition
// already stages every activation into near space itself before this pass
// would ever see it (see CIMPartition.cpp's own header) -- exactly the gap
// its summary in Passes.td already documents, the same shape as
// cim-legalize-precision's own isolation from cim-partition's output. So
// this is tested on hand-written IR that manufactures the space mismatch
// cim-partition's current output never produces.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::cim;

namespace {

bool isNearSpace(MemRefType type) {
  auto space = llvm::dyn_cast_or_null<SpaceAttr>(type.getMemorySpace());
  return space && space.getKind() == SpaceKind::near;
}

/// `type` with the same shape and element type, but #cim.space<near> --
/// the one space cim.mvm's activation operand is required to live in.
MemRefType nearVariant(MemRefType type) {
  return MemRefType::get(type.getShape(), type.getElementType(),
                         /*layout=*/MemRefLayoutAttrInterface{},
                         SpaceAttr::get(type.getContext(), SpaceKind::near));
}

/// The region `value` is defined in: a block argument's owning block's
/// region, or an op result's parent region. Uniform handling of both is
/// what lets isLoopInvariant below treat a loop's own induction variable
/// (a block argument of the loop body) the same as any other
/// loop-body-computed value -- neither is safe to hoist.
Region *homeRegion(Value value) {
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(value))
    return blockArg.getOwner()->getParent();
  return value.getDefiningOp()->getParentRegion();
}

/// True iff `value` is defined entirely outside `loop` -- not in its body,
/// and not in any region nested inside that body -- so a single copy made
/// once before the loop reads the same bytes on every iteration.
bool isLoopInvariant(Value value, scf::ForOp loop) {
  return !loop.getRegion().isAncestor(homeRegion(value));
}

struct CIMInsertTransfersPass
    : public CIMInsertTransfersBase<CIMInsertTransfersPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Keyed by (source value, anchor op immediately before which the copy
    // is inserted): every mvm sharing the same activation source at the
    // same hoist point reuses one cim.copy instead of duplicating it. The
    // anchor is the enclosing scf.for when the source is loop-invariant
    // with respect to it, or the mvm itself otherwise -- that difference is
    // exactly what makes hoisting and non-hoisting share one code path.
    llvm::DenseMap<std::pair<Value, Operation *>, Value> copies;

    WalkResult result = module.walk([&](MvmOp op) -> WalkResult {
      auto actType = llvm::dyn_cast<MemRefType>(op.getActivations().getType());
      if (!actType) {
        op.emitError(
            "cim-insert-transfers: cim.mvm activations must be a ranked "
            "memref; this pass has no unranked-memref layout to copy from");
        return WalkResult::interrupt();
      }
      if (isNearSpace(actType))
        return WalkResult::advance();

      Value source = op.getActivations();
      Operation *anchor = op.getOperation();
      if (auto loop = op->getParentOfType<scf::ForOp>())
        if (isLoopInvariant(source, loop))
          anchor = loop.getOperation();

      auto key = std::make_pair(source, anchor);
      auto it = copies.find(key);
      if (it == copies.end()) {
        OpBuilder builder(anchor);
        auto copy =
            builder.create<CopyOp>(op.getLoc(), nearVariant(actType), source);
        it = copies.try_emplace(key, copy.getResult()).first;
      }
      op.getActivationsMutable().set(it->second);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMInsertTransfersPass() {
  return std::make_unique<CIMInsertTransfersPass>();
}
