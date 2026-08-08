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
// 1. Straight-line code only. Every cim op must sit directly in its
//    function's entry block; any cim op nested inside a region (an
//    scf.for body, in practice -- cim-placement's own loop hoisting is
//    exactly what would put one there) is refused with a diagnostic. A
//    buffer this pass allocates for one iteration would either leak on
//    every subsequent iteration or need a hoisting analysis of its own;
//    neither is designed yet.
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

    // Validate before rewriting anything: every cim op must sit directly in
    // this block. A cim op nested in some other region (checked via a full
    // walk, so it is caught regardless of how deeply nested) gets a clear
    // diagnostic instead of being silently skipped by the block-only loop
    // below and left as a dangling, stale-typed op for the verifier to
    // report confusingly once this pass has already rewritten everything
    // around it.
    LogicalResult nestingCheck = success();
    func.walk([&](Operation *op) {
      if (op->getBlock() == &body || !isCimDialectOp(op))
        return;
      op->emitError(
          "cim-lower-to-target: this v0.1 slice only lowers straight-line "
          "code directly in a function's entry block; this op is nested "
          "inside a region (e.g. an scf.for body -- exactly what "
          "cim-placement's loop hoisting produces), which has no lowering "
          "yet: a buffer this pass allocated would either leak every "
          "iteration or need a hoisting analysis of its own, neither "
          "designed here");
      nestingCheck = failure();
    });
    if (failed(nestingCheck))
      return failure();

    // make_early_inc_range: several op handlers below erase the op they are
    // visiting, which would invalidate a plain iterator over the same
    // block.
    for (Operation &opRef : llvm::make_early_inc_range(body))
      if (failed(lowerOp(&opRef)))
        return failure();
    return success();
  }

private:
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

  //===--------------------------------------------------------------===//
  // Small builders
  //===--------------------------------------------------------------===//

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
  Value hostPointer(OpBuilder &b, Location loc, Value memrefValue) {
    Value idx =
        b.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memrefValue);
    Value asI64 = b.create<arith::IndexCastOp>(loc, i64Ty, idx);
    return b.create<LLVM::IntToPtrOp>(loc, ptrTy, asI64);
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
    Value buf = allocBuffer(b, loc, dev, *bytes, space);
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
              memref::DeallocOp>(user))
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
    if (auto o = dyn_cast<RequantizeOp>(op))
      return lowerRequantize(o);
    if (auto o = dyn_cast<memref::DeallocOp>(op))
      return lowerDealloc(o);
    if (auto o = dyn_cast<memref::SubViewOp>(op))
      return lowerSubview(o);
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
    Value buf = allocBuffer(b, loc, dev, *bytes, CimrtSpace::kInsitu);
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
    Value outBuf = allocBuffer(b, loc, dev, *resultBytes, CimrtSpace::kNear);
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
      auto realAlloc = b.create<memref::AllocOp>(loc, resultType);
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
    Value outBuf = allocBuffer(b, loc, dev, *resultBytes, stageSpace);

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
      auto realAlloc = b.create<memref::AllocOp>(loc, resultType);
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
        auto realAlloc = b.create<memref::AllocOp>(loc, resultType);
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

    auto fn = getOrInsertFunc(
        "cimrt_reduce_add",
        b.getFunctionType({ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i32Ty}, {i32Ty}));
    Value countVal = constI64(b, loc, count);
    Value bitsVal = constI32(b, loc, static_cast<int32_t>(bits));

    bool accScratch = false;
    Value acc = stageForRead(b, loc, dev, partials[0], stageSpace, accScratch);
    for (size_t i = 1; i < partials.size(); ++i) {
      bool rhsScratch = false;
      Value rhs = stageForRead(b, loc, dev, partials[i], stageSpace, rhsScratch);
      Value sum = allocBuffer(b, loc, dev, *bytes, stageSpace);
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
      auto realAlloc = b.create<memref::AllocOp>(loc, destType);
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
      Value buf = allocBuffer(b, loc, dev, bytes, space);
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
      auto realAlloc = b.create<memref::AllocOp>(loc, destType);
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
    Value dstBuf = allocBuffer(b, loc, dev, bytes, space);
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
    if (operand.getType() != ptrTy)
      return success(); // a real memref: an ordinary dealloc, untouched.
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
    Value sliced = allocBuffer(b, loc, dev, range.length, space);
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
