//===- CIMLegalizePrecision.cpp - Pass 6: requantize legalization -*- C++ -*-//
//
// Spec Sec. 6, Pass 6. Inserts cim.requantize after every "terminal" i32
// accumulator -- a cim.reduce_partial result, or a cim.mvm result that no
// cim.reduce_partial consumes (the single-K-tile case, where there is
// nothing to reduce). "Terminal" is the operative word: an mvm feeding a
// reduce_partial is an intermediate partial sum, not yet the value a
// consumer outside the CIM domain should ever see.
//
// scale=1.0, zero_point=0, always. v0.1 has no per-layer calibration step
// anywhere in this pipeline to derive anything else from -- there is no
// scale/zero_point in the target schema (docs/target-format.md's
// `precision:` section has only output_effective_bits), and inventing
// calibration data this pass has no way to validate would be exactly the
// "silently produce a wrong-but-plausible number" this project's passes
// refuse to do elsewhere (see lib/Interpreter/Interpreter.cpp's own stated
// contract). What this pass legalizes is the OP SHAPE spec Sec. 6 calls
// for -- every accumulator requantized before leaving the CIM domain -- and
// the CLAMP that output_effective_bits genuinely does encode: on a target
// declaring fewer than 8 effective bits (an analog ADC's real resolution
// limit), some encodings the i8 result type could otherwise hold are not
// physically reachable, and cim.requantize (spec Sec. 5.3) exists
// specifically to make that loss explicit in the IR rather than silent.
// Real scale/zero_point derivation from an actual calibration step is spec
// M4 ("cim-legalize-precision with real effective_bits modeling",
// docs/roadmap.md) -- separate work from what lands here.
//
// NOT YET WIRED INTO THE LIVE PIPELINE, same as cim-insert-transfers's own
// stub note: cim-partition always emits its own cim.copy back to a host
// buffer typed to match the ORIGINAL i32 accumulator, with no knowledge
// that a later pass might requantize that value down to i8. Running this
// pass after cim-partition on real pipeline output changes a terminal
// value's type out from under that already-emitted cim.copy, which the
// dialect's verifier correctly rejects ("copy must not change element
// type") rather than silently coercing. Fixing that needs cim-partition
// (or a dataflow pass between the two) to know a downstream requantize is
// coming and size its own host-side buffer for i8, which is a real,
// separate piece of work -- not attempted here. This pass is complete and
// tested in isolation, on IR shaped the way cim-partition's output IS
// shaped locally (a terminal cim.mvm/cim.reduce_partial result and
// nothing else assuming its old type); test/Transforms/cim-partition.mlir
// plus test/Transforms/cim-legalize-precision.mlir is the seam where that
// integration work will eventually connect the two passes for real.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Target/TargetSpec.h"
#include "cim/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::cim;

namespace {

/// True iff `value` already has a cim.requantize reading it -- this pass is
/// idempotent, so re-running it (e.g. because a caller's pipeline includes
/// it twice) must not stack a second requantize on top of the first.
bool alreadyLegalized(Value value) {
  for (Operation *user : value.getUsers())
    if (isa<RequantizeOp>(user))
      return true;
  return false;
}

/// True iff any cim.reduce_partial consumes `value` -- i.e. `value` is an
/// intermediate K-tile partial sum, not yet the value a consumer outside
/// the CIM domain should see.
bool consumedByReducePartial(Value value) {
  for (Operation *user : value.getUsers())
    if (isa<ReducePartialOp>(user))
      return true;
  return false;
}

/// `type` with the same shape and memory space, but an i8 element type --
/// this project's v0.1 INT8 contract applies to requantized output exactly
/// as it does to weights and activations (docs/abstraction.md).
MemRefType i8Like(MemRefType type) {
  return MemRefType::get(type.getShape(), IntegerType::get(type.getContext(), 8),
                         /*layout=*/MemRefLayoutAttrInterface{},
                         type.getMemorySpace());
}

struct CIMLegalizePrecisionPass
    : public CIMLegalizePrecisionBase<CIMLegalizePrecisionPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (targetYAML.empty()) {
      module.emitError(
          "cim-legalize-precision requires -target-yaml=<path>; the clamp "
          "range cim.requantize enforces comes from the target's "
          "precision.output_effective_bits (spec Sec. 7)");
      return signalPassFailure();
    }
    ::cim::TargetSpec spec;
    std::string error;
    if (!::cim::parseTargetSpecFromFile(targetYAML, spec, &error)) {
      module.emitError("cim-legalize-precision: ") << error;
      return signalPassFailure();
    }
    const uint32_t effectiveBits = spec.precision.outputEffectiveBits;

    // Collected before any rewriting: inserting a cim.requantize adds a new
    // user of its own input, which consumedByReducePartial/alreadyLegalized
    // must not see while still deciding about OTHER terminal values in the
    // same walk.
    SmallVector<Value> terminals;
    module.walk([&](ReducePartialOp op) { terminals.push_back(op.getResult()); });
    module.walk([&](MvmOp op) {
      if (!consumedByReducePartial(op.getResult()))
        terminals.push_back(op.getResult());
    });

    for (Value terminal : terminals) {
      if (alreadyLegalized(terminal))
        continue;
      auto inputType = dyn_cast<MemRefType>(terminal.getType());
      if (!inputType) {
        terminal.getDefiningOp()->emitError(
            "cim-legalize-precision: terminal accumulator is not a memref");
        return signalPassFailure();
      }

      Operation *definingOp = terminal.getDefiningOp();
      OpBuilder builder(definingOp);
      builder.setInsertionPointAfter(definingOp);
      auto requantize = builder.create<RequantizeOp>(
          definingOp->getLoc(), i8Like(inputType), terminal,
          /*scale=*/builder.getF32FloatAttr(1.0f),
          /*zero_point=*/builder.getI32IntegerAttr(0),
          /*effective_bits=*/builder.getI32IntegerAttr(
              static_cast<int32_t>(effectiveBits)));
      terminal.replaceAllUsesExcept(requantize.getResult(), requantize);

      if (effectiveBits < 8) {
        const uint32_t levelsLost =
            (1u << 8) - (1u << effectiveBits);
        definingOp->emitWarning(
            "cim-legalize-precision: this target's precision."
            "output_effective_bits (")
            << effectiveBits << ") is less than 8; " << levelsLost
            << " of the 256 encodings an i8 result could otherwise hold "
               "are not reachable through this readout path (modeling "
               "analog ADC resolution loss, spec Sec. 5.3) -- accuracy "
               "impact scales with how much of the value range those "
               "encodings covered, and is not estimated further here";
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMLegalizePrecisionPass() {
  return std::make_unique<CIMLegalizePrecisionPass>();
}
