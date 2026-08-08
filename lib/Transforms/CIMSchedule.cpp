//===- CIMSchedule.cpp - Pass 4: order emitted cim ops ----------*- C++ -*-===//
//
// v0.1 scope (spec Sec. 6, Pass 4): keep source order and insert cim.barrier
// conservatively. Nothing is reordered or overlapped -- that is v0.2's
// double-buffering work, gated on the target's capabilities.double_buffer_
// program (docs/target-format.md) -- so the only question this pass answers
// is *where* a barrier must sit.
//
// THE RULE: cim.program and cim.mvm are the two ops spec Sec. 3.3/5.3 call
// out as the ones that actually cost time on real hardware; every other op
// keeps accumulating as an open, unsynchronized run PER DEVICE until
// something actually reads one of that run's results -- directly as an
// operand, or from inside a nested region (an scf.for body referencing a
// resident defined outside the loop is exactly this: the loop op itself has
// no such operand, only something nested in its region does). At that
// point, and only for the device(s) actually depended on, a cim.barrier is
// inserted immediately before the dependent op -- never after, since
// inserting it after would let that op race the work it is reading.
// Unrelated bookkeeping (a fresh memref.alloc, a tile_alloc on a different
// device, a loop's own bound computation) passes through untouched: it does
// not depend on anything pending, so nothing needs to wait for it and it
// does not need to wait either.
//
// A block's outstanding work is also always fully flushed right before that
// block's last operation (its terminator, for every block this pass
// actually reaches -- func.return or scf.yield): nothing outside a block
// can be assumed to wait for it otherwise. For an scf.for body this is what
// gives a hoisted cim.program's dependents a fresh barrier on EVERY runtime
// iteration, not just once total, without this pass needing to know
// anything about trip counts (contrast cim-placement, spec Sec. 6 Pass 3,
// which very much does).
//
// This is genuinely conservative, not merely cautious: it never reorders
// anything (source order is preserved exactly), and a barrier that turns
// out not to have been strictly necessary is still correct, only not
// maximally overlapped -- which v0.1 never claims to be.
//
// Scheduling is per-block and blind to its parent/children: this pass walks
// every block in the module (module.walk visits every Region of every
// Operation, so an scf.for's body block is scheduled exactly like a
// function's), and nested dependency detection (via Operation::walk on each
// candidate op) is what lets a parent block's scheduling correctly "see"
// into a child region without this pass needing any bespoke recursion.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Transforms/Passes.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::cim;

namespace {

/// The !cim.device value `op` (a ProgramOp or MvmOp) issues work to, or
/// failure if that cannot be traced -- which cim-partition's output never
/// triggers (every cim.program's tile always comes from cim.tile_alloc, and
/// every cim.mvm's weights always come from cim.program), but a malformed
/// or hand-written module could. A device op whose device cannot be traced
/// is treated like any other non-device op by the caller: it cannot be
/// accumulated into a per-device run, so it can only ever trigger a flush
/// of something else's pending work, never itself become pending.
FailureOr<Value> deviceFor(Operation *op) {
  if (auto program = dyn_cast<ProgramOp>(op)) {
    auto tileAlloc = program.getTile().getDefiningOp<TileAllocOp>();
    if (!tileAlloc)
      return failure();
    return tileAlloc.getDevice();
  }
  if (auto mvm = dyn_cast<MvmOp>(op)) {
    auto program = mvm.getWeights().getDefiningOp<ProgramOp>();
    auto tileAlloc =
        program ? program.getTile().getDefiningOp<TileAllocOp>() : nullptr;
    if (!tileAlloc)
      return failure();
    return tileAlloc.getDevice();
  }
  return failure();
}

/// True iff `op`, or anything nested inside a region it owns, reads any
/// value in `results` as an operand. The nested-region case is not a
/// hypothetical: an scf.for op's own operand list is just its bounds and
/// iter_args, never a value only referenced by ops inside its body, so a
/// hoisted cim.program's resident being consumed by a cim.mvm inside the
/// loop is invisible to a check of the scf.for op's direct operands alone.
bool dependsOnAny(Operation *op, const llvm::DenseSet<Value> &results) {
  bool found = false;
  op->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands()) {
      if (results.contains(operand)) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

/// Schedule one block: walk it in source order, tracking one open
/// (un-barriered) run of results per device. Whenever an op -- device work
/// on a DIFFERENT device, or anything else -- depends on an open run, flush
/// that run (a cim.barrier immediately before the dependent op) before
/// proceeding. The block's own last op always flushes everything still
/// open, since nothing outside this block can be assumed to wait for it.
void scheduleBlock(Block &block) {
  if (block.empty())
    return;

  // Insertion-order map: device -> results produced by its open run so
  // far. Order matters only for making output deterministic (first device
  // touched gets its barrier emitted first when several flush together).
  llvm::MapVector<Value, llvm::DenseSet<Value>> pending;

  // early_inc_range: this loop inserts cim.barrier ops into `block` as it
  // goes (always strictly before the op currently being visited, i.e.
  // behind the iteration cursor), so the newly inserted ops are never
  // themselves visited.
  for (Operation &op : llvm::make_early_inc_range(block)) {
    FailureOr<Value> ownDevice = deviceFor(&op);
    const bool isBlockEnd = (&op == &block.back());

    llvm::SmallVector<Value> toFlush;
    for (auto &entry : pending) {
      // Continuing this device's own already-open run does not need to
      // wait on itself.
      if (succeeded(ownDevice) && entry.first == *ownDevice)
        continue;
      if (isBlockEnd || dependsOnAny(&op, entry.second))
        toFlush.push_back(entry.first);
    }
    if (!toFlush.empty()) {
      OpBuilder builder(&op);
      for (Value device : toFlush) {
        builder.create<BarrierOp>(op.getLoc(), device);
        pending.erase(device);
      }
    }

    if (succeeded(ownDevice)) {
      llvm::DenseSet<Value> &results = pending[*ownDevice];
      for (Value result : op.getResults())
        results.insert(result);
    }
  }
}

struct CIMSchedulePass : public CIMScheduleBase<CIMSchedulePass> {
  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          scheduleBlock(block);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMSchedulePass() {
  return std::make_unique<CIMSchedulePass>();
}
