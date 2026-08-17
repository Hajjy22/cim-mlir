//===- CIMLowerToTarget.cpp - Pass 7: lower to cimrt calls -----*- C++ -*-===//
//
// Spec Sec. 6, Pass 7. Converts cim ops into real calls against cimrt.h's
// actual C ABI -- func.call to extern "C" cimrt_* declarations -- so the
// result can go through MLIR's standard --convert-to-llvm pipeline,
// mlir-translate, and a linker, and come out as a real binary. Nothing else
// in this project has needed to CALL cimrt from generated code before now:
// the interpreter (lib/Interpreter/Interpreter.cpp) executes cim ops by
// calling cimrt directly from C++, one op at a time, as it walks the IR. This
// pass is that same job done at compile time instead of run time, and reuses
// the interpreter's own staging model (see below) because it is the same
// problem.
//
// v0.1 SCOPE -- READ BEFORE EXTENDING THIS PASS:
//
// 1. Straight-line code, plus arbitrarily many levels of PLAIN scf.for
//    nesting. Every cim op must sit directly in its function's entry
//    block, or directly in the body of a plain (no iter_args) scf.for
//    that itself sits in that entry block or in the body of another such
//    loop, recursively -- exactly the shape cim-placement's own loop
//    hoisting produces at its one supported level once a matmul does not
//    fit entirely resident (spec Sec. 6): the resident-programming ops it
//    can hoist stay straight-line before the loop, and the per-row
//    activation-stage/mvm/readback ops that cannot are left in the loop
//    body. cim-placement itself never emits a SECOND level of nesting
//    (v0.1 handles one, docs/roadmap.md M3) -- the loop support below one
//    level deep exists for IR this pass can still be handed directly
//    (hand-written, or from a future placement pass that does nest), the
//    same way cim.reduce_partial's and a genuine memref.subview's lowering
//    (points 2 below) exist for shapes cim-partition already emits today
//    even though earlier revisions of this pass refused them too. Only ONE
//    kind of nesting has a lowering here: a plain scf.for inside another
//    plain scf.for, arbitrarily deep. Anything else nested inside a
//    recognized loop -- an scf.if, or a loop with iter_args, at ANY depth
//    -- is still refused with a diagnostic: a loop-carried device value
//    would need to survive as part of that loop's own iter_args type list,
//    which this pass' rewriting (and cimrt's own buffer-handle model) says
//    nothing about, and conditional control flow has no notion here of
//    which branch a hoisted buffer's single allocation should apply to.
//
//    The hoisting analysis this pass has generalizes to any depth by
//    hoisting to just before the OUTERMOST recognized loop in the nest,
//    not merely the nearest enclosing one: every buffer this pass
//    allocates while lowering a recognized loop nest's body
//    (loopHoistBefore, below, which -- once set on entering the
//    OUTERMOST loop -- is left unchanged while recursing into any loop
//    nested inside it) is created ONCE, immediately before that outermost
//    loop, instead of at the un-lowered op's own position inside it --
//    every loop body in the nest keeps only the write/compute/read calls
//    that use the result, which is an ordinary SSA value any body in the
//    nest can reference with no special capture (scf.for's region has no
//    capture-list concept to begin with, and a value defined before the
//    outermost loop dominates every level nested inside it). This is what
//    keeps a real, multi-row (or multi-row-and-column) matmul from leaking
//    one buffer's worth of device memory per innermost iteration: the SAME
//    handle is reused (overwritten) on every iteration of every nesting
//    level, matching how actual scratchpad hardware would behave, rather
//    than a fresh one being handed out and then abandoned. A buffer's
//    matching free (either this pass' own cimrt_free for a device handle,
//    or a real memref.dealloc for a host one) is, symmetrically, deferred
//    to just after the OUTERMOST loop, the same as its allocation, if --
//    and only if -- that specific buffer's own allocation was
//    itself hoisted (hoistedThisLoop, below): freeing the one shared,
//    reused buffer on whichever iteration the original free happened to
//    sit on would make every later iteration's use of it a use-after-
//    free. A buffer this pass allocates AND frees within the same op's
//    own lowering (e.g. stageForRead's scratch buffer, cim.program's
//    weight-staging buffer, reduce_partial's non-final accumulators) is
//    hoisted exactly the same way, for the same reason -- reused every
//    iteration instead of realloc'd -- its free just moves with it.
// 2. cim.reduce_partial IS lowered (lowerReducePartial, below), against a
//    new cimrt_reduce_add ABI call that sums two device buffers elementwise
//    -- an N-operand reduce_partial becomes N-1 chained calls, matching
//    Interpreter.cpp's runReducePartial's own left-to-right fold. This was
//    originally refused here with "needs its own buffer-lifetime story
//    this pass does not have (multiple partial-sum buffers alive at once,
//    reduced into one)" -- that turned out to overstate the problem, the
//    same way cim.requantize's original refusal did (see its own note
//    below): the N partial-sum buffers are not fresh scratch this pass
//    would need to juggle, they are ALREADY-live cim.mvm results this pass
//    lowered earlier in the same straight-line walk, exactly like any
//    other device-space value that outlives its producing op (file header
//    point 4) -- reduce_partial's only new bookkeeping is the N-2
//    INTERMEDIATE accumulator buffers lowerReducePartial itself allocates
//    to chain the adds, which it frees as it goes, keeping only the final
//    one alive as the op's result.
//
//    cim.requantize IS also lowered (lowerRequantize, below), against a new
//    cimrt_requantize ABI call that does the round-half-away-from-zero and
//    signed-clamp arithmetic device-side -- this pass has never computed a
//    value itself anywhere else (it stages memory and dispatches calls;
//    cimrt does the real work), and reproducing that arithmetic a second
//    time in generated IR would risk it silently drifting from
//    Interpreter.cpp's own copy. cim-legalize-precision makes a
//    requantize's input its terminal accumulator's SOLE consumer, so
//    there is no multi-buffer lifetime puzzle here the way there is for
//    reduce_partial -- one producer, one consumer, exactly like an mvm's
//    staged activation. The one wrinkle: cimrt_requantize needs the
//    input's element width independently of the (always-safe, own) result
//    type, and a device-space input operand that was already lowered by an
//    earlier op in this same pass is just a bare !llvm.ptr with no width
//    of its own -- see deviceValueElemBits below, for exactly the same
//    reason tileDevices exists.
//
//    A genuine (non-identity) memref.subview of a device-space value IS
//    also now lowered, when it is a rank-1, unit-stride slice of a rank-1
//    source -- exactly what cim-partition's subView1D emits slicing a
//    staged activation once a matmul spans more than one K-tile (spec
//    Sec. 6). This used to be refused outright alongside cim.reduce_partial
//    and cim.requantize, for the same reason cimrt_mvm has no offset or
//    sub-buffer concept: there was no ABI call that could express "give me
//    a fresh buffer holding bytes [offset, offset+length) of this other
//    buffer." There is now: cimrt_copy_range (runtime/include/cimrt.h), a
//    byte-range generalization of cimrt_copy the same way cimrt_write/
//    cimrt_read's own offset parameters generalize a whole-buffer
//    transfer. checkAllowedConsumers computes the byte range while the
//    subview's SOURCE type is still real and records it in
//    materializedSliceRange (below) for lowerSubview to use once it no
//    longer can (see that map's own comment); an IDENTITY slice (offset 0,
//    full extent) still folds through with no new buffer, exactly as
//    before -- this is strictly additive. A higher-rank source, a
//    non-unit stride, or a dynamic offset/size is still refused: those
//    have no cimrt_copy_range equivalent (one contiguous byte range) and
//    are a real, separate ABI decision (M4 in docs/roadmap.md), not
//    something to guess at here. Closing this is what lets a REAL
//    multi-K-tile matmul from cim-partition reach cimrt_mvm through this
//    pass at all -- before, it could only ever reach the first K-tile's
//    identity slice before tripping this refusal.
// 3. Memory model: a #cim.space<near|insitu> memref is NOT a real memref
//    after this pass runs -- cimrt's buffers are opaque handles (you
//    cannot get a raw pointer into device memory the way memref-to-llvm
//    assumes), so every such SSA value is instead replaced end-to-end by
//    the !llvm.ptr (a cimrt_buffer*) that cimrt_alloc returned for it.
//    #cim.space<host> (or unspaced) memrefs are untouched real memrefs;
//    their bytes reach cimrt only via cimrt_write/cimrt_read, extracting a
//    raw pointer with memref.extract_aligned_pointer_as_index.
//
//    Rewriting happens strictly in program order and every op is erased as
//    it is lowered, so a downstream op's operand is ALREADY the new value
//    (an !llvm.ptr, an i32 tile id, ...) by the time this pass looks at it
//    -- Value::replaceAllUsesWith did that when the producing op was
//    lowered. That is what makes an operand's CURRENT type self-describing:
//    checking whether e.g. cim.mvm's activations operand is still a
//    MemRefType or already ptrTy tells this pass, with no separate
//    bookkeeping, whether it is looking at real host memory or an already-
//    staged device buffer (the common case on real pipeline output --
//    cim-partition already stages activations into near space itself, see
//    its own file header). Only a genuinely host-side operand gets staged
//    into a fresh scratch buffer, written, and freed once its one
//    consuming call returns, exactly like the interpreter's own
//    runProgram/runMvm. Operand-type inspection cannot recover two
//    things: which DEVICE a bare i32 tile id belongs to (an integer
//    carries no such link -- tileDevices, below), and what element WIDTH
//    an already-lowered !llvm.ptr device value used to have (a pointer
//    carries no width -- deviceValueElemBits, below, needed by
//    lowerRequantize and by a materialized subview's own bookkeeping).
//    A third map, materializedSliceRange, closes a similar gap for a
//    genuine subview's byte range specifically (see point 2 above and its
//    own comment). Those three maps are the only real bookkeeping this
//    pass keeps.
// 4. Freeing: this pass frees only the SCRATCH buffers it allocates
//    itself purely to stage one call (weight/activation staging in
//    lowerProgram/lowerMvm). A device-space value that OUTLIVES its
//    producing op (an mvm result kept as a handle, a copy's destination)
//    is freed only if a later memref.dealloc on it appears in the IR --
//    translated here to cimrt_free -- never inferred. A device-space
//    value with no such dealloc anywhere leaks, exactly as it would in
//    hand-written C using cimrt directly with no free. No liveness
//    analysis exists here to do better; this is a known limitation, not
//    an oversight.
// 5. Errors: cimrt_status is never ignored. Every call's result is routed
//    through cf.assert -- there is no interpreter here to catch a
//    wrong-but-plausible number, so a real compiled program traps rather
//    than silently continuing on a runtime failure.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Target/TargetSpec.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::cim;

namespace {

/// cimrt_space (cimrt.h). Not the dialect's SpaceKind -- this is the C ABI's
/// own enum, whose values are part of the ABI and must match cimrt.h
/// exactly.
enum CimrtSpace : int32_t { kHost = 0, kNear = 1, kInsitu = 2 };

/// True iff `op` belongs to the cim dialect -- what this pass' straight-line
/// nesting check and op dispatch both key on.
bool isCimDialectOp(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && dialect->getNamespace() == CIMDialect::getDialectNamespace();
}

/// True iff a cim op sits anywhere inside `op`'s own nested regions (not
/// counting `op` itself) -- what distinguishes "this op is itself a cim op,
/// nothing to check" from "this op is some OTHER construct that has a cim
/// op buried inside it", the two cases collectRecognizedLoopBodies (below)
/// must tell apart.
bool containsCimOpStrictly(Operation *op) {
  bool found = false;
  op->walk([&](Operation *nested) {
    if (nested != op && isCimDialectOp(nested))
      found = true;
  });
  return found;
}

/// Recursively validates, and collects into `loopBodies`, every "recognized"
/// scf.for loop body reachable from `block` -- a plain loop (no iter_args)
/// containing at least one cim op somewhere inside it, at any depth of
/// nesting inside other such loops. This is file header point 1's
/// straight-line-plus-arbitrarily-many-levels-of-plain-scf.for shape, and
/// the two ways it can fail are exactly the two call sites below: an
/// scf.for with iter_args whose body contains a cim op (no lowering exists
/// for a loop-carried device value), or any OTHER kind of region -- most
/// obviously an scf.if -- that contains a cim op at any depth (no lowering
/// exists for a cim op inside conditional control flow at all). Both
/// return `failure` with a diagnostic already emitted on the offending op;
/// the caller (FunctionLowering::run) should stop rather than continue
/// once this returns failure.
///
/// An op in `block` that is itself a cim op is skipped (it has no regions
/// to recurse into, and is none of this function's concern -- lowerOp
/// dispatches it directly). An op that is neither a cim op nor CONTAINS one
/// anywhere inside it (an unrelated scf.if, an unrelated scf.for with
/// nothing cim-related in its body, a plain arith op, ...) is also none of
/// this function's concern and is left alone, exactly as before this
/// function existed.
LogicalResult
collectRecognizedLoopBodies(Block &block,
                            llvm::DenseSet<Block *> &loopBodies) {
  for (Operation &opRef : block) {
    if (isCimDialectOp(&opRef))
      continue;
    if (!containsCimOpStrictly(&opRef))
      continue;
    auto forOp = dyn_cast<scf::ForOp>(&opRef);
    if (!forOp) {
      opRef.emitError(
          "cim-lower-to-target: a cim op nested inside a construct other "
          "than a plain scf.for (e.g. an scf.if) has no lowering in this "
          "v0.1 slice");
      return failure();
    }
    if (!forOp.getInitArgs().empty()) {
      forOp.emitError(
          "cim-lower-to-target: an scf.for with loop-carried values "
          "(iter_args) whose body contains a cim op has no lowering in "
          "this v0.1 slice -- only a plain counting loop with no carried "
          "state is supported");
      return failure();
    }
    loopBodies.insert(forOp.getBody());
    if (failed(collectRecognizedLoopBodies(*forOp.getBody(), loopBodies)))
      return failure();
  }
  return success();
}

/// Whole bytes for one element of `type`, or failure for anything not a
/// positive multiple of 8 bits -- this pass is exactly as byte-addressed as
/// the interpreter it mirrors (see Interpreter.cpp's elementByteWidth) and
/// for the same reason: a truncated-to-zero byte width would silently
/// compute a zero-size allocation.
FailureOr<int64_t> elementByteWidth(Type elementType) {
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType || intType.getWidth() == 0 || intType.getWidth() % 8 != 0)
    return failure();
  return static_cast<int64_t>(intType.getWidth() / 8);
}

/// Total byte size of a statically-shaped memref, or failure for a dynamic
/// shape or sub-byte element type -- this pass has no dynamic-size story,
/// matching the rest of the v0.1 pipeline's static-shape contract.
FailureOr<int64_t> byteSizeOf(MemRefType type) {
  if (!type.hasStaticShape())
    return failure();
  FailureOr<int64_t> elemBytes = elementByteWidth(type.getElementType());
  if (failed(elemBytes))
    return failure();
  return type.getNumElements() * *elemBytes;
}

/// True iff `view` slices its entire source with zero offset, unit stride,
/// and no rank reduction -- i.e. it is a no-op view, not a genuine slice.
/// Every field is checked against its STATIC value; a dynamic offset, size,
/// or stride (a runtime operand this function cannot see the value of)
/// always fails this check rather than being assumed to be the identity --
/// this is the "must be verified, not assumed" case the identity-subview
/// fold exists to get right (see checkAllowedConsumers below).
///
/// Must only be called while `view`'s source operand is still a real
/// memref: it uses SubViewOp's own typed accessors (getSourceType(), which
/// is built on the same kind of unconditional cast as the ODS accessors
/// lowerTileAlloc's comment warns about), which are unsafe once this pass
/// has rewritten that operand to something else. checkAllowedConsumers,
/// the only caller, always runs before any such rewriting happens.
bool isIdentitySubview(memref::SubViewOp view) {
  MemRefType srcType = view.getSourceType();
  ArrayRef<int64_t> srcShape = srcType.getShape();
  // No rank reduction: static_offsets/sizes/strides always have one entry
  // per SOURCE dimension regardless of whether the result is rank-reduced
  // (a rank-reducing subview just drops some of those dimensions from the
  // RESULT's own shape), so the only direct way to rule that out is
  // comparing the result's actual declared rank to the source's.
  if (view.getType().getRank() != static_cast<int64_t>(srcShape.size()))
    return false;
  ArrayRef<int64_t> offsets = view.getStaticOffsets();
  ArrayRef<int64_t> sizes = view.getStaticSizes();
  ArrayRef<int64_t> strides = view.getStaticStrides();
  if (offsets.size() != srcShape.size() || sizes.size() != srcShape.size() ||
      strides.size() != srcShape.size())
    return false;
  for (int64_t off : offsets)
    if (off != 0)
      return false;
  for (int64_t stride : strides)
    if (stride != 1)
      return false;
  for (auto [size, dim] : llvm::zip_equal(sizes, srcShape)) {
    if (ShapedType::isDynamic(size) || ShapedType::isDynamic(dim) ||
        size != dim)
      return false;
  }
  return true;
}

/// A static byte offset and length into some buffer.
struct ByteRange {
  int64_t offset = 0;
  int64_t length = 0;
};

/// The (byte offset, byte length) a rank-1, unit-stride, statically
/// shaped-and-offset memref.subview describes, computed from the
/// subview's own RESULT type alone -- deliberately not the SOURCE type,
/// which may already be ptrTy by the time lowerSubview needs this (file
/// header point 3): a device value's element width has no home once
/// rewritten, but a memref.subview op's own declared RESULT type is never
/// rewritten, only its OPERAND is, so this is safe to call both before
/// AND after that rewrite -- unlike isIdentitySubview and the source-rank
/// check in checkAllowedConsumers, which need the SOURCE type and must
/// only run before it. Fails on a dynamic offset/stride/shape or a
/// non-unit stride (a genuinely strided gather has no cimrt_copy_range
/// equivalent, which moves one contiguous byte range).
FailureOr<ByteRange> rank1ContiguousSliceByteRange(memref::SubViewOp view) {
  MemRefType resultType = view.getType();
  if (resultType.getRank() != 1 || !resultType.hasStaticShape())
    return failure();
  FailureOr<int64_t> elemBytes = elementByteWidth(resultType.getElementType());
  if (failed(elemBytes))
    return failure();
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(resultType, strides, offset)) ||
      ShapedType::isDynamic(offset) || strides.size() != 1 ||
      strides[0] != 1)
    return failure();
  return ByteRange{offset * *elemBytes, resultType.getShape()[0] * *elemBytes};
}

/// Lowers one function's straight-line cim ops to cimrt_* calls. One
/// instance per function so tileDevices never leaks across functions that
/// do not share SSA values.
class FunctionLowering {
public:
  FunctionLowering(ModuleOp m, const ::cim::TargetSpec &targetSpec,
                    unsigned &globalCounter)
      : module(m), ctx(m.getContext()), spec(targetSpec),
        stringGlobalCounter(globalCounter) {
    ptrTy = LLVM::LLVMPointerType::get(ctx);
    i32Ty = IntegerType::get(ctx, 32);
    i64Ty = IntegerType::get(ctx, 64);
    i1Ty = IntegerType::get(ctx, 1);
    f32Ty = Float32Type::get(ctx);
  }

  LogicalResult run(func::FuncOp func) {
    if (!func.getBody().hasOneBlock())
      return func.emitError(
          "cim-lower-to-target: only single-block function bodies are "
          "supported in this v0.1 slice");
    Block &body = func.getBody().front();

    // Recognize, and validate, every level of plain scf.for nesting around
    // a cim op, at any depth -- see file header point 1 for why, and
    // collectRecognizedLoopBodies's own comment for exactly what it
    // accepts and refuses. `loopBodies` ends up containing every
    // recognized loop's body block, at every nesting level, not just the
    // top ones directly in `body`.
    llvm::DenseSet<Block *> loopBodies;
    if (failed(collectRecognizedLoopBodies(body, loopBodies)))
      return failure();

    // Validate before rewriting anything: every cim op must sit directly
    // in this block, or in the body of one of the recognized loops
    // above (at any depth -- loopBodies already contains every level). A
    // cim op anywhere else (checked via a full walk, so it is caught
    // regardless of how deeply nested) gets a clear diagnostic instead of
    // being silently skipped by the dispatch loop below and left as a
    // dangling, stale-typed op for the verifier to report confusingly
    // once this pass has already rewritten everything around it.
    LogicalResult nestingCheck = success();
    func.walk([&](Operation *op) {
      if (!isCimDialectOp(op))
        return;
      if (op->getBlock() == &body || loopBodies.contains(op->getBlock()))
        return;
      op->emitError(
          "cim-lower-to-target: this v0.1 slice only lowers straight-line "
          "code directly in a function's entry block, or in the body of "
          "a plain scf.for loop nested (at any depth) in that block; this "
          "op is inside some OTHER construct not covered by that shape "
          "(e.g. an scf.if, or a loop with iter_args), which has no "
          "lowering yet");
      nestingCheck = failure();
    });
    if (failed(nestingCheck))
      return failure();

    // make_early_inc_range: several op handlers below erase the op they are
    // visiting, which would invalidate a plain iterator over the same
    // block.
    for (Operation &opRef : llvm::make_early_inc_range(body)) {
      if (auto forOp = dyn_cast<scf::ForOp>(&opRef);
          forOp && loopBodies.contains(forOp.getBody())) {
        if (failed(lowerLoopBody(forOp, loopBodies)))
          return failure();
        continue;
      }
      if (failed(lowerOp(&opRef)))
        return failure();
    }
    return success();
  }

private:
  /// Lowers every op directly in a recognized loop's body, with
  /// loopHoistBefore set so every buffer this pass creates along the way
  /// is hoisted to just before `forOp` instead of built at its own (in-
  /// loop) position -- see that field's own comment. `forOp` itself is
  /// left in the IR untouched (it is not a cim op; lowerOp's dispatch has
  /// no case for it and nothing here erases it) -- only its body's ops
  /// are rewritten, exactly like the entry block's own ops are.
  ///
  /// Recurses into a nested scf.for that collectRecognizedLoopBodies
  /// already proved recognized (present in `loopBodies`) -- exactly the
  /// same membership test the top-level dispatch loop in run() uses for
  /// the outermost loop, so a nested for-loop that is NOT in loopBodies
  /// (one with no cim op anywhere inside it -- collectRecognizedLoopBodies
  /// never adds such a loop) is left to lowerOp's default case instead,
  /// i.e. untouched, exactly as it would be if it sat at the top level.
  /// `loopHoistBefore` is set once, on entering the OUTERMOST loop in the
  /// nest, and left unchanged for the rest of the recursion -- see file
  /// header point 1 for why every level shares the same hoist point.
  LogicalResult lowerLoopBody(scf::ForOp forOp,
                              const llvm::DenseSet<Block *> &loopBodies) {
    const bool isOutermost = loopHoistBefore == nullptr;
    if (isOutermost) {
      loopHoistBefore = forOp;
      hoistedThisLoop.clear();
    }
    LogicalResult result = success();
    for (Operation &opRef : llvm::make_early_inc_range(*forOp.getBody())) {
      if (auto innerFor = dyn_cast<scf::ForOp>(&opRef);
          innerFor && loopBodies.contains(innerFor.getBody())) {
        if (failed(lowerLoopBody(innerFor, loopBodies))) {
          result = failure();
          break;
        }
        continue;
      }
      if (failed(lowerOp(&opRef))) {
        result = failure();
        break;
      }
    }
    if (isOutermost) {
      loopHoistBefore = nullptr;
      hoistedThisLoop.clear();
    }
    return result;
  }

  ModuleOp module;
  MLIRContext *ctx;
  const ::cim::TargetSpec &spec;
  unsigned &stringGlobalCounter;

  Type ptrTy, i32Ty, i64Ty, i1Ty, f32Ty;

  /// The i32 tile-id constant this pass materialized for a cim.tile_alloc
  /// (and reused verbatim for the cim.program that programs it) -> the
  /// !llvm.ptr device it was allocated on. The one piece of bookkeeping an
  /// operand's own type cannot recover -- see file header point 3.
  llvm::DenseMap<Value, Value> tileDevices;
  /// A device-space !llvm.ptr this pass handed back as the replacement for
  /// some memref-typed value -> that memref's own element bit width.
  /// Needed only by lowerRequantize, which -- unlike every other lowering
  /// here -- needs an INPUT operand's element width even when that operand
  /// is already ptrTy (an earlier mvm/copy this same pass lowered staged
  /// it): a bare pointer carries no width of its own, the same kind of gap
  /// tileDevices exists to close for tile ids. Populated everywhere a
  /// device-space handle is created and kept alive as a value's
  /// replacement (lowerMvm, lowerCopy, lowerRequantize's own device-space
  /// branch) -- see file header point 3.
  llvm::DenseMap<Value, unsigned> deviceValueElemBits;
  /// Every device this pass has opened in this function, in program order.
  /// cim.copy carries no device operand of its own in the dialect (its ODS
  /// arguments are just `$source`), so a call that needs one to route
  /// cimrt_alloc/cimrt_write/cimrt_read/cimrt_copy through uses "any device
  /// opened so far" -- the exact same rule the interpreter's own
  /// runCimCopy uses, and for the same reason: cim-partition's real output
  /// never opens more than one.
  llvm::SmallVector<Value> openDevices;
  /// A memref.subview op proved by checkAllowedConsumers to be a genuine
  /// (non-identity) rank-1 contiguous slice of a device-space value ->
  /// the byte range it describes. Populated there because computing it
  /// needs the SOURCE type's rank, which checkAllowedConsumers can still
  /// see (the source is still real at that point) but lowerSubview cannot
  /// (file header point 3; see rank1ContiguousSliceByteRange's own
  /// comment). Absence of an entry for a given op means checkAllowedConsumers
  /// proved it was the IDENTITY slice instead: lowerSubview folds that case
  /// straight through to its source handle with no new buffer, exactly as
  /// it did before this map existed.
  llvm::DenseMap<Operation *, ByteRange> materializedSliceRange;
  /// Non-null exactly while lowering the body of a recognized scf.for
  /// nest this pass is processing (run()/lowerLoopBody), set to the
  /// OUTERMOST scf.for in that nest -- and left pointing there for the
  /// whole recursion into any loop nested inside it, however many levels
  /// deep -- see file header point 1 for why a loop needs this at all,
  /// and why every level shares the same one. creationBuilder() is what
  /// actually redirects a buffer-creation call site to build immediately
  /// before this op instead of at its own (in-loop) position; freeBuffer
  /// and lowerDealloc consult hoistedThisLoop (below) to decide whether a
  /// given buffer's free must move too.
  Operation *loopHoistBefore = nullptr;
  /// Every buffer (a device !llvm.ptr from allocBuffer, or a host memref
  /// from memref::AllocOp) this pass has itself created via
  /// creationBuilder()/noteHoisted() while lowering the CURRENT loop nest
  /// (every level of it, not just whichever level is being visited right
  /// now). Cleared on entry to, and exit from, the OUTERMOST
  /// lowerLoopBody call -- see loopHoistBefore's own comment for why this
  /// exists.
  llvm::DenseSet<Value> hoistedThisLoop;

  //===--------------------------------------------------------------===//
  // Small builders
  //===--------------------------------------------------------------===//

  /// The builder a buffer-creation call site (allocBuffer, or a host
  /// memref.alloc this pass creates for a readback buffer) should
  /// actually build with: `atOp` unchanged in straight-line code, or one
  /// positioned immediately before the enclosing scf.for when lowering a
  /// recognized loop body -- see loopHoistBefore's own comment. Every
  /// value such a call site passes in when building (a byte size, a
  /// space kind, a device handle already itself hoisted or opened before
  /// the loop, ...) is already loop-invariant by construction, so
  /// hoisting the whole creation sequence changes nothing about what
  /// value it computes, only when, and how often, it runs. The caller
  /// must pass the Value it goes on to create to noteHoisted() right
  /// after creating it, so freeBuffer/lowerDealloc can later recognize it.
  OpBuilder creationBuilder(OpBuilder &atOp) {
    if (!loopHoistBefore)
      return atOp;
    return OpBuilder(loopHoistBefore);
  }

  /// Records that `buf` was just created via a creationBuilder() this
  /// pass returned -- a no-op in straight-line code, since
  /// hoistedThisLoop only matters while loopHoistBefore is set. Always
  /// safe to call unconditionally right after any buffer creation.
  void noteHoisted(Value buf) {
    if (loopHoistBefore)
      hoistedThisLoop.insert(buf);
  }

  func::FuncOp getOrInsertFunc(StringRef name, FunctionType type) {
    if (auto fn = module.lookupSymbol<func::FuncOp>(name))
      return fn;
    OpBuilder b(ctx);
    b.setInsertionPointToStart(module.getBody());
    auto fn = b.create<func::FuncOp>(module.getLoc(), name, type);
    fn.setPrivate();
    return fn;
  }

  Value constI32(OpBuilder &b, Location loc, int32_t v) {
    return b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(v));
  }
  Value constI64(OpBuilder &b, Location loc, int64_t v) {
    return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(v));
  }

  /// A raw !llvm.ptr into `memrefValue`'s own storage. Only ever called on a
  /// value whose type is still a real MemRefType (see file header point 3);
  /// an already-staged device buffer has no raw pointer to extract.
  ///
  /// memref.extract_aligned_pointer_as_index deliberately returns ONLY the
  /// underlying allocation's base pointer -- verified directly against real
  /// --expand-strided-metadata/--finalize-memref-to-llvm output, which
  /// lowers it to a bare `llvm.extractvalue` of the descriptor's aligned-
  /// pointer field and NEVER adds the descriptor's separate offset field,
  /// static or dynamic. Every subview this pass or cim-partition ever
  /// builds is unit-stride (subView/subView1D/rowSubView all pass
  /// stride 1), so extract_strided_metadata's `offset` result, in
  /// elements, times the element width in bytes is exactly the missing
  /// displacement -- added here by hand so this function's callers can
  /// keep treating ANY host memref uniformly, offset or not.
  ///
  /// This closes a real, PRE-EXISTING gap: every caller of this function
  /// got away with the missing offset before M > 1 batching, because
  /// every activation subview reaching cim.copy had a literal, always-zero
  /// offset (m == 1 only), and no real-target-e2e binary happened to give
  /// cim.program's 2nd-or-later N/K-tile weightBlock (a genuinely nonzero
  /// static offset already) numerically distinguishable data (every such
  /// test's weight is uniform -- see e.g. check-multi-k-tile.mlir.in's
  /// all-ones weight). A batched matmul with a real, distinguishable
  /// per-row activation is what finally made a wrong address produce a
  /// wrong, checkable number instead of a lucky match.
  Value hostPointer(OpBuilder &b, Location loc, Value memrefValue) {
    auto memrefType = cast<MemRefType>(memrefValue.getType());
    auto metadata =
        b.create<memref::ExtractStridedMetadataOp>(loc, memrefValue);
    Value baseIdx = b.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, metadata.getBaseBuffer());
    Value baseI64 = b.create<arith::IndexCastOp>(loc, i64Ty, baseIdx);
    const int64_t elemBytes = memrefType.getElementTypeBitWidth() / 8;
    Value elemBytesVal = b.create<arith::ConstantIndexOp>(loc, elemBytes);
    Value offsetBytes =
        b.create<arith::MulIOp>(loc, metadata.getOffset(), elemBytesVal);
    Value offsetBytesI64 = b.create<arith::IndexCastOp>(loc, i64Ty, offsetBytes);
    Value addr = b.create<arith::AddIOp>(loc, baseI64, offsetBytesI64);
    return b.create<LLVM::IntToPtrOp>(loc, ptrTy, addr);
  }

  /// %status == CIMRT_OK, or trap with `what`. Every cimrt_* call's result
  /// is routed through this -- see file header point 5.
  void checkOk(OpBuilder &b, Location loc, Value status, StringRef what) {
    Value zero = constI32(b, loc, 0);
    Value ok = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, status,
                                       zero);
    b.create<cf::AssertOp>(loc, ok, what);
  }

  /// A single !llvm.ptr-sized stack slot, for the handful of cimrt calls
  /// that hand a value back through an out-parameter (cimrt_open,
  /// cimrt_alloc) rather than returning it directly.
  Value allocaSlot(OpBuilder &b, Location loc) {
    Value one = constI64(b, loc, 1);
    return b.create<LLVM::AllocaOp>(loc, ptrTy, ptrTy, one);
  }

  Value allocBuffer(OpBuilder &b, Location loc, Value dev, int64_t bytes,
                    CimrtSpace space) {
    auto fn = getOrInsertFunc(
        "cimrt_alloc", b.getFunctionType({ptrTy, i64Ty, i32Ty, ptrTy}, {i32Ty}));
    Value slot = allocaSlot(b, loc);
    Value size = constI64(b, loc, bytes);
    Value spaceVal = constI32(b, loc, space);
    Value status =
        b.create<func::CallOp>(loc, fn, ValueRange{dev, size, spaceVal, slot})
            .getResult(0);
    checkOk(b, loc, status, "cimrt_alloc failed");
    return b.create<LLVM::LoadOp>(loc, ptrTy, slot);
  }

  void freeBuffer(OpBuilder &b, Location loc, Value buf) {
    auto fn = getOrInsertFunc("cimrt_free", b.getFunctionType({ptrTy}, {}));
    if (loopHoistBefore && hoistedThisLoop.contains(buf)) {
      // This buffer's own allocation was itself hoisted to before the
      // loop (see loopHoistBefore's own comment) -- its free must move
      // with it, to just after the loop, not stay at this call's own
      // position inside the loop body: freeing the one shared, reused
      // buffer on whichever iteration reaches this call would make every
      // later iteration's use of it a use-after-free.
      OpBuilder afterLoop(loopHoistBefore->getContext());
      afterLoop.setInsertionPointAfter(loopHoistBefore);
      afterLoop.create<func::CallOp>(loc, fn, ValueRange{buf});
      return;
    }
    b.create<func::CallOp>(loc, fn, ValueRange{buf});
  }

  void writeBuffer(OpBuilder &b, Location loc, Value buf, Value hostPtr,
                   int64_t bytes) {
    auto fn = getOrInsertFunc(
        "cimrt_write", b.getFunctionType({ptrTy, i64Ty, ptrTy, i64Ty}, {i32Ty}));
    Value zero = constI64(b, loc, 0);
    Value size = constI64(b, loc, bytes);
    Value status =
        b.create<func::CallOp>(loc, fn, ValueRange{buf, zero, hostPtr, size})
            .getResult(0);
    checkOk(b, loc, status, "cimrt_write failed");
  }

  void readBuffer(OpBuilder &b, Location loc, Value buf, Value hostPtr,
                  int64_t bytes) {
    auto fn = getOrInsertFunc(
        "cimrt_read", b.getFunctionType({ptrTy, i64Ty, ptrTy, i64Ty}, {i32Ty}));
    Value zero = constI64(b, loc, 0);
    Value size = constI64(b, loc, bytes);
    Value status =
        b.create<func::CallOp>(loc, fn, ValueRange{buf, zero, hostPtr, size})
            .getResult(0);
    checkOk(b, loc, status, "cimrt_read failed");
  }

  /// A device-space handle for `memrefOperand`, for a call about to read it
  /// (activations, a copy's device-space source): if `memrefOperand` is
  /// already ptrTy (an earlier mvm/copy this pass lowered staged it),
  /// reuse it directly, no new allocation -- the common case on real
  /// pipeline output. Otherwise it is still a real host memref: stage a
  /// FRESH scratch buffer of `space`, write its bytes in, and report it as
  /// scratch (the caller must free it once its one use is done) via
  /// `isScratch`.
  Value stageForRead(OpBuilder &b, Location loc, Value dev,
                     Value memrefOperand, CimrtSpace space, bool &isScratch) {
    if (memrefOperand.getType() == ptrTy) {
      isScratch = false;
      return memrefOperand;
    }
    auto type = cast<MemRefType>(memrefOperand.getType());
    FailureOr<int64_t> bytes = byteSizeOf(type);
    assert(succeeded(bytes) && "checked by the caller before this is reached");
    OpBuilder allocB = creationBuilder(b);
    Value buf = allocBuffer(allocB, loc, dev, *bytes, space);
    noteHoisted(buf);
    Value ptr = hostPointer(b, loc, memrefOperand);
    writeBuffer(b, loc, buf, ptr, *bytes);
    isScratch = true;
    return buf;
  }

  /// True iff every user of `deviceValue` (still memref-typed, about to be
  /// replaced by a !llvm.ptr) is an op this pass knows how to lower for that
  /// use. Called right before the replacement, so an op with no defined
  /// lowering here is reported clearly instead of silently becoming
  /// ill-typed IR that only the post-pass verifier would catch, confusingly.
  ///
  /// A memref.subview user is allowed two ways. An IDENTITY slice
  /// (isIdentitySubview) -- offset 0, full extent, unit stride, so the
  /// "slice" carves out the whole buffer -- folds straight through to the
  /// same handle once lowerSubview reaches it, no new buffer (see that
  /// function). A genuine rank-1, unit-stride slice of a rank-1 source --
  /// exactly what cim-partition's subView1D emits slicing a staged
  /// activation per K-tile once there is more than one (spec Sec. 6) -- is
  /// instead MATERIALIZED into a fresh buffer via a new cimrt_copy_range
  /// call (runtime/include/cimrt.h), since cimrt_mvm itself has no
  /// offset/sub-buffer concept of its own; its byte range is recorded in
  /// materializedSliceRange for lowerSubview to use once it can no longer
  /// see the source type. Anything else -- a higher-rank source, a
  /// non-unit stride, a dynamic offset/size -- is refused with its own
  /// diagnostic: a genuinely strided or multi-dimensional slice has no
  /// cimrt_copy_range equivalent (it moves one contiguous byte range),
  /// which is an actual ABI decision this v0.1 slice does not make
  /// (tracked as M4 in docs/roadmap.md), not a guess here.
  LogicalResult checkAllowedConsumers(Value deviceValue) {
    for (Operation *user : deviceValue.getUsers()) {
      if (isa<ProgramOp, MvmOp, CopyOp, RequantizeOp, ReducePartialOp,
              ReduceMaxOp, memref::DeallocOp>(user))
        continue;
      if (auto view = dyn_cast<memref::SubViewOp>(user)) {
        if (isIdentitySubview(view))
          continue;
        if (view.getSourceType().getRank() == 1) {
          if (FailureOr<ByteRange> range = rank1ContiguousSliceByteRange(view);
              succeeded(range)) {
            materializedSliceRange[user] = *range;
            continue;
          }
        }
        return user->emitError(
            "cim-lower-to-target: this memref.subview of a "
            "#cim.space<near|insitu> value is neither an identity slice "
            "(offset 0, full extent, unit stride) nor a supported rank-1 "
            "contiguous slice of a rank-1 source (static offset, static "
            "size, unit stride) -- cimrt_copy_range moves one contiguous "
            "byte range and cimrt_mvm has no offset/sub-buffer concept of "
            "its own, so any other slice shape has no lowering in this "
            "v0.1 slice (see the M4 roadmap entry)");
      }
      return user->emitError(
          "cim-lower-to-target: this op consumes a #cim.space<near|insitu> "
          "value with no lowering defined for that use in this v0.1 slice "
          "(only cim.program, cim.mvm, cim.copy, cim.requantize, "
          "cim.reduce_partial, memref.dealloc and a supported "
          "memref.subview consuming a device-space value are supported)");
    }
    return success();
  }

  //===--------------------------------------------------------------===//
  // Op lowering
  //===--------------------------------------------------------------===//

  LogicalResult lowerOp(Operation *op) {
    if (auto o = dyn_cast<DeviceOpenOp>(op))
      return lowerDeviceOpen(o);
    if (auto o = dyn_cast<TileAllocOp>(op))
      return lowerTileAlloc(o);
    if (auto o = dyn_cast<TileFreeOp>(op))
      return lowerTileFree(o);
    if (auto o = dyn_cast<ProgramOp>(op))
      return lowerProgram(o);
    if (auto o = dyn_cast<MvmOp>(op))
      return lowerMvm(o);
    if (auto o = dyn_cast<CopyOp>(op))
      return lowerCopy(o);
    if (auto o = dyn_cast<BarrierOp>(op))
      return lowerBarrier(o);
    if (auto o = dyn_cast<ReducePartialOp>(op))
      return lowerReducePartial(o);
    if (auto o = dyn_cast<ReduceMaxOp>(op))
      return lowerReduceMax(o);
    if (auto o = dyn_cast<RequantizeOp>(op))
      return lowerRequantize(o);
    if (auto o = dyn_cast<memref::DeallocOp>(op))
      return lowerDealloc(o);
    if (auto o = dyn_cast<memref::SubViewOp>(op))
      return lowerSubview(o);

    // A cim.* op with no case above is a REAL gap, not something to pass
    // through. Everything around it gets rewritten into cimrt calls, so a
    // survivor would be left dangling among lowered code and reach
    // mlir-translate as an unknown dialect op -- failing far from its
    // cause, or worse, being silently dropped by a later pass.
    //
    // Every op in the dialect has a dispatch case above as of
    // cim.reduce_max's own lowerReduceMax (this file), which was the last
    // one missing (PR A had deliberately scoped it to cim-run only; see
    // that op's own doc comment in CIMOps.td). This fallthrough is
    // therefore UNEXERCISED BY CONSTRUCTION today -- test/Transforms/
    // cim-lower-to-target-unhandled-cim-op.mlir, which used to pin exactly
    // this path via cim.reduce_max, was deleted for that reason (see
    // docs/roadmap.md's cim.reduce_max compiled-lowering entry). It stays
    // written as a dialect check rather than a per-op list precisely so
    // the NEXT op added to the dialect inherits the same protection
    // without anyone having to remember this file or re-add a test for it
    // -- until that day, reaching this branch means a new op was added to
    // CIMOps.td without a matching case here, not a real user-facing gap.
    // (lowerReduceMax has its OWN, narrower refusal for a specific operand
    // SHAPE it cannot yet stage -- a non-contiguous memref.subview -- which
    // is a different thing from having no lowering at all; see its own
    // doc comment.)
    if (op->getDialect() == op->getContext()->getLoadedDialect<CIMDialect>())
      return op->emitError(
                 "cim-lower-to-target has no lowering for this op, so it "
                 "would survive into the translated output unlowered: ")
             << op->getName()
             << " (the cim-run interpreter may still support it; this is "
                "the compiled real-target path only)";

    // Not a cim op and not memref.dealloc/memref.subview: none of this
    // pass' concern (memref.alloc, memref.get_global, arith.constant, a
    // cim_print_* call, ...) -- left exactly as it was.
    return success();
  }

  LogicalResult lowerDeviceOpen(DeviceOpenOp op) {
    if (!isa<DeviceType>(op.getResult().getType()))
      return op.emitError("cim-lower-to-target: internal error: "
                          "cim.device_open result is not !cim.device");
    OpBuilder b(op);
    Location loc = op.getLoc();

    std::string globalName =
        ("__cim_target_str_" + Twine(stringGlobalCounter++)).str();
    SmallVector<char> bytes(op.getTarget().begin(), op.getTarget().end());
    bytes.push_back('\0'); // cimrt_open takes a C string.
    auto strType = MemRefType::get(static_cast<int64_t>(bytes.size()),
                                   b.getI8Type());
    SmallVector<APInt> elements;
    elements.reserve(bytes.size());
    for (char c : bytes)
      elements.push_back(APInt(8, static_cast<uint64_t>(c)));
    auto tensorType =
        RankedTensorType::get(strType.getShape(), b.getI8Type());
    auto initial = DenseElementsAttr::get(tensorType, elements);
    {
      OpBuilder moduleBuilder(ctx);
      moduleBuilder.setInsertionPointToStart(module.getBody());
      moduleBuilder.create<memref::GlobalOp>(
          module.getLoc(), b.getStringAttr(globalName),
          /*sym_visibility=*/b.getStringAttr("private"),
          TypeAttr::get(strType), initial,
          /*constant=*/b.getUnitAttr(), /*alignment=*/IntegerAttr());
    }
    Value str = b.create<memref::GetGlobalOp>(loc, strType, globalName);
    Value strPtr = hostPointer(b, loc, str);

    Value slot = allocaSlot(b, loc);
    auto fn =
        getOrInsertFunc("cimrt_open", b.getFunctionType({ptrTy, ptrTy}, {i32Ty}));
    Value status =
        b.create<func::CallOp>(loc, fn, ValueRange{strPtr, slot}).getResult(0);
    checkOk(b, loc, status, "cimrt_open failed");
    Value dev = b.create<LLVM::LoadOp>(loc, ptrTy, slot);

    openDevices.push_back(dev);
    op.getResult().replaceAllUsesWith(dev);
    op.erase();
    return success();
  }

  LogicalResult lowerTileAlloc(TileAllocOp op) {
    // op.getDevice() is not safe to call here: it is an ODS-generated
    // TypedValue<DeviceType> accessor that casts unconditionally (an
    // assertion in a debug build, undefined behavior in release), and this
    // operand has already been rewritten to !llvm.ptr by the time a
    // preceding cim.device_open was lowered -- exactly the case this
    // function needs to detect, not crash on. The raw, untyped operand
    // access below is what makes that detection possible at all.
    Value dev = op->getOperand(0);
    if (isa<DeviceType>(dev.getType()))
      return op.emitError(
          "cim-lower-to-target: cim.tile_alloc's device was never opened "
          "by a cim.device_open this pass lowered (e.g. a !cim.device "
          "function argument, which has no lowering in this v0.1 slice)");
    const uint64_t id = op.getId();
    if (id >= spec.tiles.count)
      return op.emitError("cim-lower-to-target: tile id ")
             << id << " is outside the target's " << spec.tiles.count
             << " tiles (checked at compile time against -target-yaml, "
                "rather than deferring to cimrt_query at run time)";

    OpBuilder b(op);
    Value idConst = constI32(b, op.getLoc(), static_cast<int32_t>(id));
    tileDevices[idConst] = dev;
    op.getResult().replaceAllUsesWith(idConst);
    op.erase();
    return success();
  }

  LogicalResult lowerTileFree(TileFreeOp op) {
    // No cimrt call: a "tile" has no runtime object of its own to free (see
    // the file header's tile/resident note), matching the interpreter's own
    // no-op handling of this op.
    op.erase();
    return success();
  }

  LogicalResult lowerProgram(ProgramOp op) {
    // Raw operand access, not op.getTile() -- see lowerTileAlloc's comment;
    // the same asserting-cast hazard applies to every ODS-typed accessor
    // for a dialect-fixed type (CIM_DeviceType, CIM_TileType,
    // CIM_ResidentType) once this pass has rewritten that operand.
    Value tileId = op->getOperand(0);
    if (isa<TileType>(tileId.getType()))
      return op.emitError("cim-lower-to-target: cim.program's tile did not "
                          "come from a cim.tile_alloc this pass lowered");
    auto tileDevIt = tileDevices.find(tileId);
    if (tileDevIt == tileDevices.end())
      return op.emitError(
          "cim-lower-to-target: internal error: could not recover "
          "cim.program's device for its tile id");
    Value dev = tileDevIt->second;

    // Raw operand access, not op.getWeights() -- see lowerTileAlloc's
    // comment; AnyMemRef-constrained operands get a TypedValue<MemRefType>
    // accessor with the exact same asserting-cast hazard as a dialect-fixed
    // type, not just CIM_DeviceType/CIM_TileType/CIM_ResidentType.
    Value weightsOperand = op->getOperand(1);
    auto weightsType = dyn_cast<MemRefType>(weightsOperand.getType());
    if (!weightsType)
      return op.emitError(
          "cim-lower-to-target: cim.program's weights must be a ranked, "
          "still-host memref (a device-space weights operand has no "
          "lowering here)");
    FailureOr<int64_t> bytes = byteSizeOf(weightsType);
    if (failed(bytes))
      return op.emitError(
          "cim-lower-to-target: cim.program's weights must have a static "
          "shape and a whole-byte element type");

    OpBuilder b(op);
    Location loc = op.getLoc();
    OpBuilder allocB = creationBuilder(b);
    Value buf = allocBuffer(allocB, loc, dev, *bytes, CimrtSpace::kInsitu);
    noteHoisted(buf);
    Value ptr = hostPointer(b, loc, weightsOperand);
    writeBuffer(b, loc, buf, ptr, *bytes);

    auto fn = getOrInsertFunc(
        "cimrt_program", b.getFunctionType({ptrTy, i32Ty, ptrTy}, {i32Ty}));
    Value status =
        b.create<func::CallOp>(loc, fn, ValueRange{dev, tileId, buf})
            .getResult(0);
    checkOk(b, loc, status, "cimrt_program failed");
    freeBuffer(b, loc, buf);

    // The resident IS its tile's id (see file header point 3): reusing the
    // SAME Value as the replacement means tileDevices already has the
    // right entry for it, with nothing new to insert. SSA already gives
    // the dialect's own invalidate-on-reprogram rule for free here: the
    // prior resident value (if any) has no remaining users once its own
    // consumers were lowered earlier in this same walk.
    op.getResult().replaceAllUsesWith(tileId);
    op.erase();
    return success();
  }

  LogicalResult lowerMvm(MvmOp op) {
    // Raw operand access, not op.getWeights() -- see lowerTileAlloc's
    // comment. cim.mvm's $weights is CIM_ResidentType, the same
    // asserting-cast hazard as cim.program's $tile.
    Value tileId = op->getOperand(0);
    if (isa<ResidentType>(tileId.getType()))
      return op.emitError("cim-lower-to-target: cim.mvm's weights did not "
                          "come from a cim.program this pass lowered");
    auto tileDevIt = tileDevices.find(tileId);
    if (tileDevIt == tileDevices.end())
      return op.emitError(
          "cim-lower-to-target: internal error: could not recover "
          "cim.mvm's device for its resident's tile id");
    Value dev = tileDevIt->second;

    // Raw operand access, not op.getActivations() -- see lowerTileAlloc's
    // comment; this is the operand most likely to already be ptrTy on real
    // pipeline output (cim-partition stages activations into near space
    // itself), so this is not a defensive-only fix.
    Value actOperand = op->getOperand(1);
    auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultType)
      return op.emitError(
          "cim-lower-to-target: cim.mvm's result must be a ranked memref");
    FailureOr<int64_t> resultBytes = byteSizeOf(resultType);
    if (failed(resultBytes))
      return op.emitError(
          "cim-lower-to-target: cim.mvm's result must have a static shape "
          "and a whole-byte element type");
    if (actOperand.getType() != ptrTy) {
      auto actType = cast<MemRefType>(actOperand.getType());
      if (failed(byteSizeOf(actType)))
        return op.emitError(
            "cim-lower-to-target: cim.mvm's activations must have a "
            "static shape and a whole-byte element type");
    }

    OpBuilder b(op);
    Location loc = op.getLoc();
    bool actScratch = false;
    Value actBuf =
        stageForRead(b, loc, dev, actOperand, CimrtSpace::kNear, actScratch);

    // The mvm always computes into a fresh near-space buffer -- cimrt_mvm
    // has no notion of writing straight to host memory, matching the
    // interpreter's own non-device-space path in runMvm.
    OpBuilder outAllocB = creationBuilder(b);
    Value outBuf = allocBuffer(outAllocB, loc, dev, *resultBytes, CimrtSpace::kNear);
    noteHoisted(outBuf);
    Value accumulate = b.create<arith::ConstantOp>(
        loc, b.getIntegerAttr(i1Ty, op.getAccumulate() ? 1 : 0));

    auto fn = getOrInsertFunc(
        "cimrt_mvm",
        b.getFunctionType({ptrTy, i32Ty, ptrTy, ptrTy, i1Ty}, {i32Ty}));
    Value status =
        b.create<func::CallOp>(
             loc, fn, ValueRange{dev, tileId, actBuf, outBuf, accumulate})
            .getResult(0);
    checkOk(b, loc, status, "cimrt_mvm failed");
    if (actScratch)
      freeBuffer(b, loc, actBuf);

    Value original = op.getResult();
    if (auto space = dyn_cast_or_null<SpaceAttr>(resultType.getMemorySpace());
        space && space.getKind() != SpaceKind::host) {
      // A device-space result: stays an opaque handle, no read-back. Check
      // BEFORE rewriting so an unsupported consumer is reported clearly
      // rather than left as ill-typed IR (file header point 3).
      if (failed(checkAllowedConsumers(original)))
        return failure();
      original.replaceAllUsesWith(outBuf);
      deviceValueElemBits[outBuf] = resultType.getElementTypeBitWidth();
    } else {
      // A host-declared result: this pass must leave real, readable bytes
      // behind for whatever consumes it (a print, a dealloc, ...).
      auto realAlloc = creationBuilder(b).create<memref::AllocOp>(loc, resultType);
      noteHoisted(realAlloc.getResult());
      Value realPtr = hostPointer(b, loc, realAlloc.getResult());
      readBuffer(b, loc, outBuf, realPtr, *resultBytes);
      freeBuffer(b, loc, outBuf);
      original.replaceAllUsesWith(realAlloc.getResult());
    }
    op.erase();
    return success();
  }

  /// Lowers cim.requantize against the new cimrt_requantize ABI call (see
  /// this file's header point 2 and runtime/include/cimrt.h). Mirrors
  /// lowerMvm's shape closely: stage the input, allocate an output buffer,
  /// call, check the status, branch on the result's declared space.
  LogicalResult lowerRequantize(RequantizeOp op) {
    // Raw operand access, not op.getInput() -- see lowerTileAlloc's
    // comment; the input is exactly the operand most likely to already be
    // ptrTy on real pipeline output (the mvm/reduce_partial this pass
    // already lowered feeding it).
    Value inputOperand = op->getOperand(0);
    auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultType)
      return op.emitError(
          "cim-lower-to-target: cim.requantize's result must be a ranked "
          "memref");
    FailureOr<int64_t> resultBytes = byteSizeOf(resultType);
    if (failed(resultBytes))
      return op.emitError(
          "cim-lower-to-target: cim.requantize's result must have a "
          "static shape and a whole-byte element type");
    const int64_t count = resultType.getNumElements();
    const unsigned outBits = resultType.getElementTypeBitWidth();

    // The input's element width, independently of the result's: self-
    // describing when the operand is still a real memref, recovered from
    // deviceValueElemBits (file header point 3) when it is already ptrTy.
    unsigned inBits;
    if (inputOperand.getType() == ptrTy) {
      auto it = deviceValueElemBits.find(inputOperand);
      if (it == deviceValueElemBits.end())
        return op.emitError(
            "cim-lower-to-target: internal error: could not recover "
            "cim.requantize's input element width");
      inBits = it->second;
    } else {
      auto inputType = cast<MemRefType>(inputOperand.getType());
      if (failed(byteSizeOf(inputType)))
        return op.emitError(
            "cim-lower-to-target: cim.requantize's input must have a "
            "static shape and a whole-byte element type");
      inBits = inputType.getElementTypeBitWidth();
    }

    // cim.requantize carries no device operand of its own in the dialect,
    // same as cim.copy -- "any device opened so far", see lowerCopy's
    // comment.
    if (openDevices.empty())
      return op.emitError(
          "cim-lower-to-target: cim.requantize needs a device to stage "
          "through, but no cim.device_open has been lowered yet in this "
          "function");
    Value dev = openDevices.front();

    // Every real cim-legalize-precision output keeps the input's own
    // memory space on the result -- both its narrowing and width-
    // preserving shapes do (see CIMLegalizePrecision.cpp's file header) --
    // so the buffers this call stages through live in whichever space the
    // RESULT declares; a host-declared result stages/reads back exactly
    // like cim.mvm's does.
    auto resultSpace = dyn_cast_or_null<SpaceAttr>(resultType.getMemorySpace());
    const bool resultIsDevice = resultSpace && resultSpace.getKind() != SpaceKind::host;
    const CimrtSpace stageSpace =
        resultIsDevice
            ? (resultSpace.getKind() == SpaceKind::insitu ? CimrtSpace::kInsitu
                                                            : CimrtSpace::kNear)
            : CimrtSpace::kNear;

    OpBuilder b(op);
    Location loc = op.getLoc();
    bool inputScratch = false;
    Value inputBuf =
        stageForRead(b, loc, dev, inputOperand, stageSpace, inputScratch);
    OpBuilder outAllocB = creationBuilder(b);
    Value outBuf = allocBuffer(outAllocB, loc, dev, *resultBytes, stageSpace);
    noteHoisted(outBuf);

    Value countVal = constI64(b, loc, count);
    Value inBitsVal = constI32(b, loc, static_cast<int32_t>(inBits));
    Value outBitsVal = constI32(b, loc, static_cast<int32_t>(outBits));
    Value scaleVal =
        b.create<arith::ConstantOp>(loc, b.getFloatAttr(f32Ty, op.getScale()));
    // Signed reinterpretation, same as RequantizeOp::verify()'s own read of
    // effective_bits: the ODS accessor for an I32Attr is unsigned, so a
    // negative zero_point in the IR would otherwise read as a very large
    // positive one.
    Value zeroPointVal =
        constI32(b, loc, static_cast<int32_t>(op.getZeroPoint()));
    Value effectiveBitsVal =
        constI32(b, loc, static_cast<int32_t>(op.getEffectiveBits()));

    auto fn = getOrInsertFunc(
        "cimrt_requantize",
        b.getFunctionType(
            {ptrTy, ptrTy, ptrTy, i64Ty, i32Ty, i32Ty, f32Ty, i32Ty, i32Ty},
            {i32Ty}));
    Value status =
        b.create<func::CallOp>(
             loc, fn,
             ValueRange{dev, inputBuf, outBuf, countVal, inBitsVal, outBitsVal,
                        scaleVal, zeroPointVal, effectiveBitsVal})
            .getResult(0);
    checkOk(b, loc, status, "cimrt_requantize failed");
    if (inputScratch)
      freeBuffer(b, loc, inputBuf);

    Value original = op.getResult();
    if (resultIsDevice) {
      // Check BEFORE rewriting so an unsupported consumer is reported
      // clearly rather than left as ill-typed IR (file header point 3).
      if (failed(checkAllowedConsumers(original)))
        return failure();
      original.replaceAllUsesWith(outBuf);
      deviceValueElemBits[outBuf] = outBits;
    } else {
      auto realAlloc = creationBuilder(b).create<memref::AllocOp>(loc, resultType);
      noteHoisted(realAlloc.getResult());
      Value realPtr = hostPointer(b, loc, realAlloc.getResult());
      readBuffer(b, loc, outBuf, realPtr, *resultBytes);
      freeBuffer(b, loc, outBuf);
      original.replaceAllUsesWith(realAlloc.getResult());
    }
    op.erase();
    return success();
  }

  /// Lowers cim.reduce_partial against the new cimrt_reduce_add ABI call
  /// (see this file's header point 2 and runtime/include/cimrt.h): an
  /// N-operand reduce becomes N-1 chained pairwise adds, matching
  /// Interpreter.cpp's runReducePartial's own left-to-right fold. Every
  /// real operand cim-partition ever produces is already a device-space
  /// !llvm.ptr by the time this runs (an earlier cim.mvm this same pass
  /// lowered) -- staged like any other read-only operand (stageForRead)
  /// for the rare case a hand-written module gives it a real memref
  /// instead. Every INTERMEDIATE accumulator this function allocates to
  /// chain the calls is freed once its one consuming call is done, exactly
  /// like lowerMvm/lowerRequantize's own scratch buffers; only the final
  /// accumulator survives, as the op's result.
  LogicalResult lowerReducePartial(ReducePartialOp op) {
    auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultType)
      return op.emitError(
          "cim-lower-to-target: cim.reduce_partial's result must be a "
          "ranked memref");
    FailureOr<int64_t> bytes = byteSizeOf(resultType);
    if (failed(bytes))
      return op.emitError(
          "cim-lower-to-target: cim.reduce_partial's result must have a "
          "static shape and a whole-byte element type");
    const int64_t count = resultType.getNumElements();
    const unsigned bits = resultType.getElementTypeBitWidth();

    // Raw operand access, not op.getPartials() -- see lowerTileAlloc's
    // comment; ReducePartialOp's Variadic<AnyMemRef> operands get the same
    // asserting-cast-typed accessor hazard as every fixed-type operand
    // this pass has already rewritten. Validated up front, all of them,
    // before any call is emitted: stageForRead asserts success on a
    // still-real-memref operand's byte size rather than checking it itself
    // (every other caller of it -- lowerMvm, lowerRequantize -- already
    // checks first for the same reason), so this loop is what makes a
    // malformed operand a clean diagnostic instead of a crash.
    SmallVector<Value> partials(op->getOperands().begin(),
                                op->getOperands().end());
    for (Value partial : partials) {
      if (partial.getType() == ptrTy)
        continue;
      auto type = dyn_cast<MemRefType>(partial.getType());
      if (!type || failed(byteSizeOf(type)))
        return op.emitError(
            "cim-lower-to-target: cim.reduce_partial's operands must be "
            "ranked memrefs with a static shape and a whole-byte element "
            "type");
    }

    // cim.reduce_partial carries no device operand of its own in the
    // dialect, same as cim.copy/cim.requantize -- "any device opened so
    // far", see lowerCopy's comment.
    if (openDevices.empty())
      return op.emitError(
          "cim-lower-to-target: cim.reduce_partial needs a device to stage "
          "through, but no cim.device_open has been lowered yet in this "
          "function");
    Value dev = openDevices.front();

    // Every real cim-partition output keeps every partial's memory space
    // consistent with the reduction's result (spec Sec. 5.4 rule 4: all
    // operands and the result share one shape and element type) -- see
    // lowerRequantize's identical reasoning for why the stage space
    // follows the RESULT's declared space, not an operand's.
    auto resultSpace = dyn_cast_or_null<SpaceAttr>(resultType.getMemorySpace());
    const bool resultIsDevice =
        resultSpace && resultSpace.getKind() != SpaceKind::host;
    const CimrtSpace stageSpace =
        resultIsDevice
            ? (resultSpace.getKind() == SpaceKind::insitu ? CimrtSpace::kInsitu
                                                            : CimrtSpace::kNear)
            : CimrtSpace::kNear;

    OpBuilder b(op);
    Location loc = op.getLoc();
    Value original = op.getResult();

    auto finish = [&](Value acc, bool accIsOwnScratch) -> LogicalResult {
      if (resultIsDevice) {
        // Check BEFORE rewriting so an unsupported consumer is reported
        // clearly rather than left as ill-typed IR (file header point 3).
        // `acc` staying alive as a handle is correct regardless of
        // accIsOwnScratch: either way it is now this value's one live
        // reference, exactly like lowerMvm/lowerRequantize's own device
        // branch.
        if (failed(checkAllowedConsumers(original)))
          return failure();
        original.replaceAllUsesWith(acc);
        deviceValueElemBits[acc] = bits;
      } else {
        auto realAlloc = creationBuilder(b).create<memref::AllocOp>(loc, resultType);
        noteHoisted(realAlloc.getResult());
        Value realPtr = hostPointer(b, loc, realAlloc.getResult());
        readBuffer(b, loc, acc, realPtr, *bytes);
        if (accIsOwnScratch)
          freeBuffer(b, loc, acc);
        original.replaceAllUsesWith(realAlloc.getResult());
      }
      op.erase();
      return success();
    };

    if (partials.size() == 1) {
      // cim-partition never emits this -- a single K-tile has "nothing to
      // reduce" and the op is not created at all (see its own file
      // header) -- but a hand-written module could, and forwarding the
      // sole operand is the only correct lowering, not a call with one
      // fake input.
      bool scratch = false;
      Value only = stageForRead(b, loc, dev, partials[0], stageSpace, scratch);
      return finish(only, scratch);
    }

    Value countVal = constI64(b, loc, count);
    Value bitsVal = constI32(b, loc, static_cast<int32_t>(bits));

    // capabilities.partial_sum_in_place (spec target-format.md): a target
    // that can accumulate without a fresh destination buffer per step gets
    // exactly ONE accumulator allocation for the whole chain instead of
    // N-1 -- see cimrt_reduce_add_inplace's own doc comment in cimrt.h for
    // why this is a separate ABI call rather than a relaxed
    // cimrt_reduce_add, and why partials[0]'s own buffer is copied into a
    // fresh one rather than mutated directly (it may have other uses this
    // pass cannot see -- stageForRead can hand back an ALREADY-LIVE handle,
    // not always a fresh one, whenever the operand is already device-space).
    if (spec.capabilities.partialSumInPlace) {
      OpBuilder accAllocB = creationBuilder(b);
      Value acc = allocBuffer(accAllocB, loc, dev, *bytes, stageSpace);
      noteHoisted(acc);

      bool firstScratch = false;
      Value first =
          stageForRead(b, loc, dev, partials[0], stageSpace, firstScratch);
      auto copyFn = getOrInsertFunc("cimrt_copy",
                                    b.getFunctionType({ptrTy, ptrTy}, {i32Ty}));
      Value copyStatus =
          b.create<func::CallOp>(loc, copyFn, ValueRange{acc, first})
              .getResult(0);
      checkOk(b, loc, copyStatus, "cimrt_copy failed");
      if (firstScratch)
        freeBuffer(b, loc, first);

      auto inplaceFn = getOrInsertFunc(
          "cimrt_reduce_add_inplace",
          b.getFunctionType({ptrTy, ptrTy, ptrTy, i64Ty, i32Ty}, {i32Ty}));
      for (size_t i = 1; i < partials.size(); ++i) {
        bool rhsScratch = false;
        Value rhs =
            stageForRead(b, loc, dev, partials[i], stageSpace, rhsScratch);
        Value status = b.create<func::CallOp>(
                             loc, inplaceFn,
                             ValueRange{dev, acc, rhs, countVal, bitsVal})
                            .getResult(0);
        checkOk(b, loc, status, "cimrt_reduce_add_inplace failed");
        if (rhsScratch)
          freeBuffer(b, loc, rhs);
      }
      return finish(acc, /*accIsOwnScratch=*/true);
    }

    auto fn = getOrInsertFunc(
        "cimrt_reduce_add",
        b.getFunctionType({ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i32Ty}, {i32Ty}));

    bool accScratch = false;
    Value acc = stageForRead(b, loc, dev, partials[0], stageSpace, accScratch);
    for (size_t i = 1; i < partials.size(); ++i) {
      bool rhsScratch = false;
      Value rhs = stageForRead(b, loc, dev, partials[i], stageSpace, rhsScratch);
      OpBuilder sumAllocB = creationBuilder(b);
      Value sum = allocBuffer(sumAllocB, loc, dev, *bytes, stageSpace);
      noteHoisted(sum);
      Value status =
          b.create<func::CallOp>(
               loc, fn, ValueRange{dev, sum, acc, rhs, countVal, bitsVal})
              .getResult(0);
      checkOk(b, loc, status, "cimrt_reduce_add failed");
      if (accScratch)
        freeBuffer(b, loc, acc);
      if (rhsScratch)
        freeBuffer(b, loc, rhs);
      acc = sum;
      accScratch = true; // every accumulator from here on is our own scratch
    }
    return finish(acc, /*accIsOwnScratch=*/true);
  }

  /// cim.reduce_max's compiled lowering: cimrt_reduce_max (spec-adjacent to
  /// cim.reduce_partial above -- see this file's header point 2 and
  /// runtime/include/cimrt.h's own cimrt_reduce_max doc comment) -- an
  /// N-operand reduce becomes N-1 chained pairwise SIGNED maxes, matching
  /// Interpreter.cpp's runReduceMax's own left-to-right fold.
  ///
  /// Deliberately just lowerReducePartial's out-of-place branch with
  /// cimrt_reduce_max substituted for cimrt_reduce_add -- there is no
  /// in-place counterpart here (no cimrt_reduce_max_inplace exists): PR A's
  /// own runtime/include/cimrt.h doc comment records that reduce_partial
  /// shipped out-of-place first too, with in-place landing later as
  /// separate, motivated work. Not duplicated by copy-paste of a helper --
  /// this function inlines the same shape directly, the same way
  /// lowerReducePartial's own two branches are each written out rather than
  /// factored, since the accumulator-threading logic is only a few lines
  /// once the shared byteSizeOf/stageForRead/allocBuffer/freeBuffer/
  /// checkOk/getOrInsertFunc/finish machinery is reused.
  ///
  /// NON-CONTIGUOUS OPERANDS ARE REFUSED, NOT MATERIALIZED -- see this
  /// function's own diagnostic below for why: byteSizeOf and hostPointer
  /// both assume a contiguous (identity-layout) memref, an invariant that
  /// holds for every operand cim-partition itself ever builds but NOT for
  /// a hand-emitted, non-unit-stride memref.subview -- exactly the shape
  /// python/cim_frontend/emit.py's own MaxPool pooling-window gather
  /// produces (Kh*Kw strided taps fed directly into cim.reduce_max, no
  /// per-tap contiguous copy -- see emit_conv_chain_module's own
  /// `pool_params` docstring). Staging such an operand as if it were
  /// contiguous would silently copy the wrong bytes: a confident wrong
  /// answer, not a crash -- exactly the failure class this project refuses
  /// to ship. So a non-identity-layout operand is refused here with a
  /// located diagnostic instead, on the same "refuse rather than guess"
  /// discipline as checkAllowedConsumers' own refusal of a non-rank-1 or
  /// non-unit-stride memref.subview consumer of a device-space value,
  /// below. This means a MaxPool-bearing module compiles and runs under
  /// the cim-run interpreter (which gathers via arbitrary strides
  /// natively) but does not yet compile on this real-target path --
  /// docs/roadmap.md and python/README.md both say so explicitly.
  LogicalResult lowerReduceMax(ReduceMaxOp op) {
    auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultType)
      return op.emitError(
          "cim-lower-to-target: cim.reduce_max's result must be a ranked "
          "memref");
    FailureOr<int64_t> bytes = byteSizeOf(resultType);
    if (failed(bytes))
      return op.emitError(
          "cim-lower-to-target: cim.reduce_max's result must have a "
          "static shape and a whole-byte element type");
    const int64_t count = resultType.getNumElements();
    const unsigned bits = resultType.getElementTypeBitWidth();

    // Raw operand access, not an ODS accessor -- see lowerReducePartial's
    // own identical comment. Every operand is validated up front, all of
    // them, before any call is emitted, INCLUDING the contiguity check
    // this function's own header comment explains: byteSizeOf/hostPointer
    // both assume an identity (unit-stride, offset-0-relative) layout, so
    // a genuinely strided operand is refused HERE, not staged and then
    // silently mis-copied.
    SmallVector<Value> operands(op->getOperands().begin(),
                                op->getOperands().end());
    for (Value operand : operands) {
      if (operand.getType() == ptrTy)
        continue;
      auto type = dyn_cast<MemRefType>(operand.getType());
      if (!type || failed(byteSizeOf(type)))
        return op.emitError(
            "cim-lower-to-target: cim.reduce_max's operands must be "
            "ranked memrefs with a static shape and a whole-byte element "
            "type");
      if (!type.getLayout().isIdentity())
        return op.emitError(
            "cim-lower-to-target: this cim.reduce_max operand is a "
            "non-contiguous (strided or offset) memref, which this pass "
            "does not stage -- byteSizeOf/hostPointer both assume an "
            "identity layout, and copying a strided view as if it were "
            "contiguous would silently read the wrong bytes rather than "
            "fail loudly. A pooling window built from strided "
            "memref.subview taps (python/cim_frontend/emit.py's own "
            "MaxPool gather) is therefore not yet supported on this "
            "compiled real-target path -- the cim-run interpreter, whose "
            "gather() walks arbitrary strides natively, still executes "
            "it. See docs/roadmap.md's cim.reduce_max entry.");
    }

    // cim.reduce_max carries no device operand of its own in the dialect,
    // same as cim.reduce_partial/cim.copy/cim.requantize -- "any device
    // opened so far", see lowerCopy's comment.
    if (openDevices.empty())
      return op.emitError(
          "cim-lower-to-target: cim.reduce_max needs a device to stage "
          "through, but no cim.device_open has been lowered yet in this "
          "function");
    Value dev = openDevices.front();

    // Same reasoning as lowerReducePartial's own identical block: the
    // stage space follows the RESULT's declared space, not an operand's.
    auto resultSpace = dyn_cast_or_null<SpaceAttr>(resultType.getMemorySpace());
    const bool resultIsDevice =
        resultSpace && resultSpace.getKind() != SpaceKind::host;
    const CimrtSpace stageSpace =
        resultIsDevice
            ? (resultSpace.getKind() == SpaceKind::insitu ? CimrtSpace::kInsitu
                                                            : CimrtSpace::kNear)
            : CimrtSpace::kNear;

    OpBuilder b(op);
    Location loc = op.getLoc();
    Value original = op.getResult();

    auto finish = [&](Value acc, bool accIsOwnScratch) -> LogicalResult {
      if (resultIsDevice) {
        if (failed(checkAllowedConsumers(original)))
          return failure();
        original.replaceAllUsesWith(acc);
        deviceValueElemBits[acc] = bits;
      } else {
        auto realAlloc = creationBuilder(b).create<memref::AllocOp>(loc, resultType);
        noteHoisted(realAlloc.getResult());
        Value realPtr = hostPointer(b, loc, realAlloc.getResult());
        readBuffer(b, loc, acc, realPtr, *bytes);
        if (accIsOwnScratch)
          freeBuffer(b, loc, acc);
        original.replaceAllUsesWith(realAlloc.getResult());
      }
      op.erase();
      return success();
    };

    if (operands.size() == 1) {
      // Mirrors lowerReducePartial's own single-operand fallback: never
      // emitted by cim-partition (a single tap has "nothing to reduce"),
      // but a hand-written module could, and forwarding the sole operand
      // is the only correct lowering.
      bool scratch = false;
      Value only = stageForRead(b, loc, dev, operands[0], stageSpace, scratch);
      return finish(only, scratch);
    }

    Value countVal = constI64(b, loc, count);
    Value bitsVal = constI32(b, loc, static_cast<int32_t>(bits));

    auto fn = getOrInsertFunc(
        "cimrt_reduce_max",
        b.getFunctionType({ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i32Ty}, {i32Ty}));

    bool accScratch = false;
    Value acc = stageForRead(b, loc, dev, operands[0], stageSpace, accScratch);
    for (size_t i = 1; i < operands.size(); ++i) {
      bool rhsScratch = false;
      Value rhs = stageForRead(b, loc, dev, operands[i], stageSpace, rhsScratch);
      OpBuilder maxAllocB = creationBuilder(b);
      Value maxed = allocBuffer(maxAllocB, loc, dev, *bytes, stageSpace);
      noteHoisted(maxed);
      Value status =
          b.create<func::CallOp>(
               loc, fn, ValueRange{dev, maxed, acc, rhs, countVal, bitsVal})
              .getResult(0);
      checkOk(b, loc, status, "cimrt_reduce_max failed");
      if (accScratch)
        freeBuffer(b, loc, acc);
      if (rhsScratch)
        freeBuffer(b, loc, rhs);
      acc = maxed;
      accScratch = true; // every accumulator from here on is our own scratch
    }
    return finish(acc, /*accIsOwnScratch=*/true);
  }

  LogicalResult lowerCopy(CopyOp op) {
    // Raw operand access, not op.getSource() -- see lowerTileAlloc's
    // comment; this is exactly the operand this function's own srcIsDevice
    // check depends on distinguishing correctly.
    Value srcOperand = op->getOperand(0);
    auto destType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!destType)
      return op.emitError(
          "cim-lower-to-target: cim.copy's result must be a ranked memref");
    const bool srcIsDevice = srcOperand.getType() == ptrTy;
    int64_t bytes = 0;
    if (!srcIsDevice) {
      auto srcType = cast<MemRefType>(srcOperand.getType());
      FailureOr<int64_t> b = byteSizeOf(srcType);
      if (failed(b))
        return op.emitError(
            "cim-lower-to-target: cim.copy must have a static shape and a "
            "whole-byte element type");
      bytes = *b;
    } else {
      FailureOr<int64_t> b = byteSizeOf(destType);
      if (failed(b))
        return op.emitError(
            "cim-lower-to-target: cim.copy must have a static shape and a "
            "whole-byte element type");
      bytes = *b;
    }

    OpBuilder b(op);
    Location loc = op.getLoc();
    auto destSpace = dyn_cast_or_null<SpaceAttr>(destType.getMemorySpace());
    const bool destIsDevice = destSpace && destSpace.getKind() != SpaceKind::host;
    Value original = op.getResult();

    if (!srcIsDevice && !destIsDevice) {
      // Host to host: no cimrt involvement, and no device needed at all --
      // a plain memref.copy into a freshly allocated real destination.
      auto realAlloc = creationBuilder(b).create<memref::AllocOp>(loc, destType);
      noteHoisted(realAlloc.getResult());
      b.create<memref::CopyOp>(loc, srcOperand, realAlloc.getResult());
      original.replaceAllUsesWith(realAlloc.getResult());
      op.erase();
      return success();
    }

    // Every remaining case touches cimrt, and needs a device to reach it
    // through -- cim.copy carries no device operand of its own in the
    // dialect, so this uses "any device opened so far" (openDevices), the
    // same rule the interpreter's own runCimCopy uses.
    if (openDevices.empty())
      return op.emitError(
          "cim-lower-to-target: cim.copy needs a device to stage through, "
          "but no cim.device_open has been lowered yet in this function");
    Value dev = openDevices.front();

    if (!srcIsDevice && destIsDevice) {
      CimrtSpace space = destSpace.getKind() == SpaceKind::insitu
                             ? CimrtSpace::kInsitu
                             : CimrtSpace::kNear;
      OpBuilder allocB = creationBuilder(b);
      Value buf = allocBuffer(allocB, loc, dev, bytes, space);
      noteHoisted(buf);
      Value ptr = hostPointer(b, loc, srcOperand);
      writeBuffer(b, loc, buf, ptr, bytes);
      if (failed(checkAllowedConsumers(original)))
        return failure();
      original.replaceAllUsesWith(buf);
      deviceValueElemBits[buf] = destType.getElementTypeBitWidth();
      op.erase();
      return success();
    }
    if (srcIsDevice && !destIsDevice) {
      auto realAlloc = creationBuilder(b).create<memref::AllocOp>(loc, destType);
      noteHoisted(realAlloc.getResult());
      Value ptr = hostPointer(b, loc, realAlloc.getResult());
      readBuffer(b, loc, srcOperand, ptr, bytes);
      original.replaceAllUsesWith(realAlloc.getResult());
      op.erase();
      return success();
    }
    // Device to device (near <-> insitu): cimrt_copy moves buffer to buffer
    // directly, no host bytes involved.
    CimrtSpace space = destSpace.getKind() == SpaceKind::insitu
                           ? CimrtSpace::kInsitu
                           : CimrtSpace::kNear;
    OpBuilder dstAllocB = creationBuilder(b);
    Value dstBuf = allocBuffer(dstAllocB, loc, dev, bytes, space);
    noteHoisted(dstBuf);
    auto fn = getOrInsertFunc("cimrt_copy",
                              b.getFunctionType({ptrTy, ptrTy}, {i32Ty}));
    Value status =
        b.create<func::CallOp>(loc, fn, ValueRange{dstBuf, srcOperand})
            .getResult(0);
    checkOk(b, loc, status, "cimrt_copy failed");
    if (failed(checkAllowedConsumers(original)))
      return failure();
    original.replaceAllUsesWith(dstBuf);
    deviceValueElemBits[dstBuf] = destType.getElementTypeBitWidth();
    op.erase();
    return success();
  }

  LogicalResult lowerBarrier(BarrierOp op) {
    // Raw operand access, not op.getDevice() -- see lowerTileAlloc's
    // comment.
    Value dev = op->getOperand(0);
    if (isa<DeviceType>(dev.getType()))
      return op.emitError(
          "cim-lower-to-target: cim.barrier's device was never opened by "
          "a cim.device_open this pass lowered");
    OpBuilder b(op);
    auto fn = getOrInsertFunc("cimrt_barrier",
                              b.getFunctionType({ptrTy}, {i32Ty}));
    Value status =
        b.create<func::CallOp>(op.getLoc(), fn, ValueRange{dev}).getResult(0);
    checkOk(b, op.getLoc(), status, "cimrt_barrier failed");
    op.erase();
    return success();
  }

  LogicalResult lowerDealloc(memref::DeallocOp op) {
    // Raw operand access, not op.getMemref() -- it is an ODS-generated
    // TypedValue<BaseMemRefType> accessor with the same asserting-cast
    // hazard as lowerTileAlloc's op.getDevice() (see its comment): this is
    // exactly the op that may be looking at an operand already rewritten
    // to !llvm.ptr by an earlier cim.mvm/cim.copy this pass lowered.
    Value operand = op->getOperand(0);
    if (operand.getType() != ptrTy) {
      // A real memref: an ordinary dealloc. If it targets a buffer this
      // pass itself hoisted out of an enclosing loop (a host readback
      // buffer allocated once, before the loop, and reused every
      // iteration -- see loopHoistBefore's own comment), it must run
      // once AFTER the loop too, not on whichever iteration reaches this
      // op in the original body: freeing the single, shared allocation
      // on that iteration would leave every later iteration's use of it
      // a use-after-free. Relocating the op is enough -- memref.dealloc
      // needs no rewriting, only to run at the right time. Anything
      // else (a real per-iteration local like cim-placement's own
      // staged-activation scratch alloc, which this pass never touches
      // at all) is left exactly where it is.
      if (loopHoistBefore && hoistedThisLoop.contains(operand))
        op->moveAfter(loopHoistBefore);
      return success();
    }
    OpBuilder b(op);
    freeBuffer(b, op.getLoc(), operand);
    op.erase();
    return success();
  }

  /// Folds an identity memref.subview of an already-lowered device-space
  /// handle straight through to that same handle, or materializes a
  /// genuine rank-1 slice into a fresh buffer via cimrt_copy_range. Raw
  /// operand access, not op.getSource() -- same asserting-cast hazard as
  /// lowerDealloc's operand access, and for the same reason: this op may
  /// be looking at an operand already rewritten to !llvm.ptr.
  ///
  /// Reaching this function with a ptrTy source at all means
  /// checkAllowedConsumers already proved, back when the source was still
  /// a real memref and its typed accessors were still safe to call, that
  /// this specific subview is either the identity (isIdentitySubview) or a
  /// supported rank-1 contiguous slice (materializedSliceRange) --
  /// checkAllowedConsumers is the only gate that ever lets a
  /// memref.subview become a consumer of a device-space value, and it
  /// refuses (failing the whole pass) on anything else before this
  /// function is ever reached. So no shape re-checking is needed here,
  /// only which of the two shapes this particular op was.
  ///
  /// A memref.subview whose source is NOT device-space (the ordinary case
  /// -- e.g. cim-partition's own slicing of a host output buffer) is left
  /// exactly as it was, same as every other op this pass does not own.
  LogicalResult lowerSubview(memref::SubViewOp op) {
    Value src = op->getOperand(0);
    if (src.getType() != ptrTy)
      return success();

    auto rangeIt = materializedSliceRange.find(op);
    if (rangeIt == materializedSliceRange.end()) {
      // The identity case: a plain, unconditional fold, no new buffer.
      op.getResult().replaceAllUsesWith(src);
      op.erase();
      return success();
    }

    // A genuine slice: cimrt_mvm (the only real consumer of this shape in
    // practice) takes whole buffers with no offset concept of its own, so
    // the slice is materialized into a fresh buffer of its own size.
    if (openDevices.empty())
      return op.emitError(
          "cim-lower-to-target: materializing this memref.subview needs a "
          "device to allocate the sliced buffer on, but no cim.device_open "
          "has been lowered yet in this function");
    Value dev = openDevices.front();

    const ByteRange range = rangeIt->second;
    MemRefType resultType = op.getType();
    auto resultSpace = dyn_cast_or_null<SpaceAttr>(resultType.getMemorySpace());
    const CimrtSpace space = (resultSpace && resultSpace.getKind() == SpaceKind::insitu)
                                  ? CimrtSpace::kInsitu
                                  : CimrtSpace::kNear;

    OpBuilder b(op);
    Location loc = op.getLoc();
    OpBuilder allocB = creationBuilder(b);
    Value sliced = allocBuffer(allocB, loc, dev, range.length, space);
    noteHoisted(sliced);
    auto fn = getOrInsertFunc(
        "cimrt_copy_range",
        b.getFunctionType({ptrTy, i64Ty, ptrTy, i64Ty, i64Ty}, {i32Ty}));
    Value dstOffset = constI64(b, loc, 0);
    Value srcOffset = constI64(b, loc, range.offset);
    Value lengthVal = constI64(b, loc, range.length);
    Value status = b.create<func::CallOp>(
                        loc, fn,
                        ValueRange{sliced, dstOffset, src, srcOffset, lengthVal})
                       .getResult(0);
    checkOk(b, loc, status, "cimrt_copy_range failed");

    // Check BEFORE rewriting so an unsupported consumer of the NEW handle
    // is reported clearly rather than left as ill-typed IR (file header
    // point 3) -- exactly the same discipline every other device-producing
    // lowering here follows.
    if (failed(checkAllowedConsumers(op.getResult())))
      return failure();
    op.getResult().replaceAllUsesWith(sliced);
    deviceValueElemBits[sliced] = resultType.getElementTypeBitWidth();
    op.erase();
    return success();
  }
};

struct CIMLowerToTargetPass
    : public CIMLowerToTargetBase<CIMLowerToTargetPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (targetYAML.empty()) {
      module.emitError(
          "cim-lower-to-target requires -target-yaml=<path>; tile ids are "
          "validated against the target's tiles.count at compile time "
          "rather than deferring to a runtime cimrt_query");
      return signalPassFailure();
    }
    ::cim::TargetSpec spec;
    std::string error;
    if (!::cim::parseTargetSpecFromFile(targetYAML, spec, &error)) {
      module.emitError("cim-lower-to-target: ") << error;
      return signalPassFailure();
    }

    unsigned stringGlobalCounter = 0;
    // Snapshot first: FunctionLowering inserts new func.func declarations
    // (the cimrt_* externs) into the module as it goes, which would
    // invalidate a live filter iterator over module.getOps<func::FuncOp>().
    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    for (func::FuncOp func : functions) {
      if (func.isExternal())
        continue; // e.g. a cim_print_* declaration -- not this pass' job.
      FunctionLowering lowering(module, spec, stringGlobalCounter);
      if (failed(lowering.run(func)))
        return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMLowerToTargetPass() {
  return std::make_unique<CIMLowerToTargetPass>();
}
