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
// COMPOSES WITH cim-partition'S OUTPUT. cim-partition always emits its own
// cim.copy back to a host buffer typed to match the ORIGINAL i32
// accumulator, with no knowledge that a later pass might requantize that
// value down to i8 -- naively rewiring that copy's operand to a narrower
// requantize result would leave its declared result type stale and trip
// the dialect verifier ("copy must not change element type"). Rather than
// teach cim-partition to anticipate a requantize that may never run (that
// would couple two otherwise-independent passes and misreport buffer sizes
// whenever this pass is absent), this pass looks at what is actually
// downstream of each terminal before deciding how to narrow it:
//
//   - If every downstream consumer of the terminal is something this pass
//     is free to retype in place (another cim.copy it can widen the
//     element type of, or a memref.copy into a scratch memref.alloc with
//     no other uses, or a memref.dealloc, which has no separate declared
//     type to go stale) -- narrow all the way to i8, this pass' normal,
//     tile-native-precision behavior, and retype that whole local chain to
//     match (collectRetypableChain / retypeChain below).
//   - Otherwise -- the common case on real cim-partition output, where the
//     copy-back's ultimate destination is a subview of the *function's own
//     output argument*, a type this pass has no license to change (that
//     would silently alter the compiled function's external signature) --
//     fall back to a requantize whose result type matches the terminal's
//     ORIGINAL element type exactly. This is not a lesser form of
//     precision legalization: cim.requantize's clamp to effective_bits is
//     a VALUE-range operation (see Interpreter.cpp's runRequantize, which
//     already supports any integer result width), so the readout-path
//     accuracy loss spec Sec. 5.3 cares about is modeled correctly either
//     way -- only the storage container's width differs, and only when
//     narrowing it would require changing a type this pass does not own.
//
// Both shapes are covered by test/Transforms/cim-legalize-precision.mlir:
// the pre-existing hand-written cases (terminal's only other use is a bare
// memref.dealloc) exercise the narrowing path, and
// test/Transforms/cim-pipeline-full.mlir exercises the width-preserving
// fallback against real cim-partition output.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Target/TargetSpec.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

/// `type` with the same shape and memory space as `type` but `elem` as its
/// element type -- the general form of i8Like, used to retype an existing
/// downstream buffer (which may not currently be i32) to match.
MemRefType withElementType(MemRefType type, Type elem) {
  return MemRefType::get(type.getShape(), elem,
                         /*layout=*/MemRefLayoutAttrInterface{},
                         type.getMemorySpace());
}

/// True iff every consumer reachable from `value` is something this pass is
/// free to retype in place (to whatever narrower type the caller chooses --
/// this function is purely structural and does not itself need to know the
/// target type), and if so, appends every value along the way that would
/// need retyping to `chain`.
///
/// Recognizes exactly the shape cim-partition's own copy-back emits (see
/// this file's header): a chain of cim.copy hops, ending in either nothing,
/// a memref.dealloc, or a memref.copy into a memref.alloc this pass owns.
/// Anything else -- a memref.copy whose destination is not a local alloc
/// (concretely: a subview of a function argument, which is what
/// cim-partition's real output looks like), or any op this function does
/// not specifically recognize -- is refused rather than guessed at,
/// matching the discipline everywhere else in this pass.
bool collectRetypableChain(Value value, SmallVectorImpl<Value> &chain) {
  for (Operation *user : value.getUsers()) {
    if (isa<memref::DeallocOp>(user))
      continue; // no separate declared type to go stale

    if (auto copy = dyn_cast<CopyOp>(user)) {
      // CopyOp has exactly one operand ($source), so `value` reaching here
      // is necessarily that source -- there is no "value is the fixed
      // destination" case to guard against, unlike memref::CopyOp below.
      chain.push_back(copy.getDest());
      if (!collectRetypableChain(copy.getDest(), chain))
        return false;
      continue;
    }

    if (auto mcopy = dyn_cast<memref::CopyOp>(user)) {
      if (mcopy.getTarget() == value)
        // `value` is being WRITTEN by this copy -- an incoming edge, not an
        // outgoing dependency to validate. Its own source is handled by
        // whichever call reached `value` as a target below; nothing further
        // to check for this particular use.
        continue;
      // `value` is this copy's source and is about to be narrowed, so its
      // target must be too. Only a fresh memref.alloc this pass can prove
      // it is free to retype qualifies -- not, for example, a subview of a
      // function argument, which is what cim-partition's real output looks
      // like and exactly the case this function must refuse.
      Value dest = mcopy.getTarget();
      if (!dest.getDefiningOp<memref::AllocOp>())
        return false;
      chain.push_back(dest);
      if (!collectRetypableChain(dest, chain))
        return false;
      continue;
    }

    return false; // unrecognized consumer: refuse rather than guess
  }
  return true;
}

/// Narrows every value in `chain` (collected by collectRetypableChain) to
/// `narrowElem`, keeping each value's own shape and memory space.
void retypeChain(ArrayRef<Value> chain, Type narrowElem) {
  for (Value v : chain)
    v.setType(withElementType(cast<MemRefType>(v.getType()), narrowElem));
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

      // Decide the requantize's result type BEFORE creating it: narrow all
      // the way to i8 (this pass' normal, tile-native-precision behavior)
      // only if every downstream consumer is something this pass is free
      // to retype to match; otherwise keep the terminal's original width
      // and let effective_bits' clamp alone carry the precision-loss
      // semantics (see this file's header for why both are correct).
      SmallVector<Value> chain;
      const bool narrows = collectRetypableChain(terminal, chain);
      Type narrowElem = IntegerType::get(inputType.getContext(), 8);
      MemRefType resultType = narrows ? i8Like(inputType) : inputType;

      OpBuilder builder(definingOp);
      builder.setInsertionPointAfter(definingOp);
      auto requantize = builder.create<RequantizeOp>(
          definingOp->getLoc(), resultType, terminal,
          /*scale=*/builder.getF32FloatAttr(1.0f),
          /*zero_point=*/builder.getI32IntegerAttr(0),
          /*effective_bits=*/builder.getI32IntegerAttr(
              static_cast<int32_t>(effectiveBits)));
      terminal.replaceAllUsesExcept(requantize.getResult(), requantize);
      if (narrows)
        retypeChain(chain, narrowElem);

      if (effectiveBits < 8) {
        const unsigned resultBits = resultType.getElementTypeBitWidth();
        const uint64_t levelsLost = (uint64_t{1} << resultBits) -
                                     (uint64_t{1} << effectiveBits);
        definingOp->emitWarning(
            "cim-legalize-precision: this target's precision."
            "output_effective_bits (")
            << effectiveBits << ") is less than 8; " << levelsLost
            << " of the " << (uint64_t{1} << resultBits)
            << " encodings the result type " << resultType.getElementType()
            << " could otherwise hold are not reachable through this "
               "readout path (modeling analog ADC resolution loss, spec "
               "Sec. 5.3) -- accuracy impact scales with how much of the "
               "value range those encodings covered, and is not estimated "
               "further here";
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMLegalizePrecisionPass() {
  return std::make_unique<CIMLegalizePrecisionPass>();
}
