//===- CIMPartition.cpp - Pass 2: tile a logical matmul --------*- C++ -*-===//
//
// Spec Sec. 6, Pass 2. Rewrites a cim-detect candidate into the cim op
// sequence that actually runs on tiles: one cim.program + cim.mvm per
// weight sub-matrix, a cim.reduce_partial tree over the contraction
// dimension, and explicit cim.copy transfers across memory spaces.
//
// This is what produces the IR cim-placement then optimizes -- until this
// pass runs there are no cim.program ops to place.
//
// Weight layout: cim.mvm computes out[r] = sum_c W[r][c] * act[c], so its
// weight matrix is output-major, [N x K]. That is exactly
// linalg.matmul_transpose_b's B operand, which is why this pass lowers that
// form directly. A plain linalg.matmul stores weights [K x N] and would
// need a transpose first; rather than silently emitting a wrong-layout
// program, such candidates are left alone with a warning.
//
// v0.1 scope, enforced rather than assumed (each unmet condition warns and
// leaves the linalg op intact, so the module stays correct and simply is
// not offloaded):
//   - buffer (memref) semantics
//
// N and K need not be exact multiples of the tile geometry: spec Sec. 6
// calls for zero-padding the ragged edge, and that is what happens here.
// When N and/or K falls short of the next tile multiple, a fresh
// memref.alloc host buffer of the padded shape is zero-filled
// (linalg.fill) and the real weight/activation data is copied into its
// top-left corner (memref.copy) before tiling proceeds exactly as in the
// exact-multiple case -- so a partially-empty tile is genuinely
// programmed with zeros in its unused rows/columns, not left holding
// whatever the physical array last held. Padding rows/columns can only
// ever contribute a 0 * x term to the MVM they take part in, so they
// cannot change any answer; the only place raggedness still shows through
// is the write-back, which copies only the `n` real output rows out of
// each block's (possibly padded) tileRows-sized result -- see
// `validRows` in partition() below. Both new host buffers are compiler
// scratch this pass allocates and never frees, matching this pipeline's
// existing convention: none of cim-partition's other scratch buffers
// (the staged activation, each block's host output copy) are freed
// either -- buffer lifetime management is out of scope for v0.1 across
// the board, not just here.
//
// M > 1 (batching): a matmul with more than one output row is tiled
// EXACTLY like the m == 1 case -- weight programming, activation staging,
// mvm, reduce_partial, write-back -- except the whole per-row sequence is
// generated once, inside a genuine `scf.for` over the M rows, instead of
// once, straight-line. This is deliberately the SAME IR SHAPE
// test/real-target/check-loop.mlir.in already proves runs correctly as a
// real binary: a hand-written scf.for whose body does one row's
// linalg.matmul_transpose_b, which cim-partition already rewrote into cim
// ops on each (textual) pass through that loop body, before this pass
// could generate such a loop itself. cim-placement's existing loop
// hoisting then lifts the loop-INVARIANT weight programming entirely
// above the generated scf.for (the same weight tiles are reprogrammed on
// every row otherwise, needlessly), and cim-lower-to-target's existing
// arbitrary-depth loop lowering hoists the per-row buffers THIS pass and
// cim-lower-to-target itself allocate while lowering ops that stay inside
// the loop body, reusing one buffer across all M iterations instead of
// leaking one per row -- none of that machinery needed to change for
// this. The one exception is the K-padding scratch buffer immediately
// below `emitRow`: a plain memref.alloc this pass writes directly rather
// than a cim op, so it is hoisted by hand here, once, before the loop,
// the same way cim-lower-to-target hoists its own.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Target/TargetSpec.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

#define DEBUG_TYPE "cim-partition"

using namespace mlir;
using namespace mlir::cim;

namespace {

/// Build a contiguous memref type in the given cim address space.
MemRefType nearMemRef(MLIRContext *ctx, ArrayRef<int64_t> shape, Type elem,
                       SpaceKind space) {
  return MemRefType::get(shape, elem, /*layout=*/MemRefLayoutAttrInterface{},
                          SpaceAttr::get(ctx, space));
}

struct CIMPartitionPass : public CIMPartitionBase<CIMPartitionPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // The tile geometry to partition against comes from the target file --
    // hardcoding it is exactly what this project exists to avoid.
    if (targetYAML.empty()) {
      module.emitError("cim-partition requires -target-yaml=<path>; the tile "
                       "geometry to partition against comes from the target "
                       "description (spec Sec. 7)");
      return signalPassFailure();
    }
    ::cim::TargetSpec spec;
    std::string error;
    if (!::cim::parseTargetSpecFromFile(targetYAML, spec, &error)) {
      module.emitError("cim-partition: ") << error;
      return signalPassFailure();
    }

    const int64_t tileRows = spec.tiles.rows;
    const int64_t tileCols = spec.tiles.cols;

    SmallVector<linalg::LinalgOp> candidates;
    module.walk([&](linalg::LinalgOp op) {
      if (op->hasAttr("cim.candidate"))
        candidates.push_back(op);
    });

    for (linalg::LinalgOp op : candidates)
      partition(op, spec, tileRows, tileCols, ctx);
  }

  /// Rewrite one candidate, or explain why it was left alone.
  void partition(linalg::LinalgOp op, const ::cim::TargetSpec &spec, int64_t tileRows,
                 int64_t tileCols, MLIRContext *ctx) {
    auto transposeB = llvm::dyn_cast<linalg::MatmulTransposeBOp>(op.getOperation());
    if (!transposeB) {
      op->emitWarning("cim-partition only lowers linalg.matmul_transpose_b "
                      "(weights in output-major [N x K] layout, matching "
                      "cim.mvm); leaving this candidate unoffloaded");
      return;
    }
    if (!op.hasPureBufferSemantics()) {
      op->emitWarning("cim-partition requires buffer semantics; run "
                      "bufferization first. Leaving this candidate unoffloaded");
      return;
    }

    Value act = op.getDpsInputs()[0];
    Value weights = op.getDpsInputs()[1];
    Value out = op.getDpsInits()[0];

    auto actType = llvm::dyn_cast<MemRefType>(act.getType());
    auto weightType = llvm::dyn_cast<MemRefType>(weights.getType());
    auto outType = llvm::dyn_cast<MemRefType>(out.getType());
    if (!actType || !weightType || !outType || actType.getRank() != 2 ||
        weightType.getRank() != 2 || outType.getRank() != 2) {
      op->emitWarning("cim-partition expects rank-2 memref operands; leaving "
                      "this candidate unoffloaded");
      return;
    }

    const int64_t m = actType.getShape()[0];
    const int64_t k = actType.getShape()[1];
    const int64_t n = weightType.getShape()[0];

    if (weightType.getShape()[1] != k || outType.getShape()[1] != n) {
      op->emitWarning("cim-partition: inconsistent matmul shapes; leaving this "
                      "candidate unoffloaded");
      return;
    }
    // Spec Sec. 6 says zero-pad a ragged edge rather than emit a partial
    // tile (which would silently compute wrong results by reading or
    // writing past the real matrix). Round each dimension up to the next
    // tile multiple; when a dimension already is one, paddedN == n /
    // paddedK == k and every branch below that only fires "when padded"
    // is simply dead, so the exact-multiple case emits identical IR to
    // before this padding support existed.
    //
    // llvm::divideCeil takes and returns uint64_t; n/k/tileRows/tileCols
    // are shape dimensions and therefore never negative (MLIR shapes are
    // non-negative by construction), so the explicit cast back to int64_t
    // here is exactly the widen-then-narrow round trip that is always
    // value-preserving for this input, not a silent truncation risk --
    // spelled out rather than left implicit, which is what
    // bugprone-narrowing-conversions is flagging.
    const int64_t paddedN =
        static_cast<int64_t>(llvm::divideCeil(n, tileRows) * tileRows);
    const int64_t paddedK =
        static_cast<int64_t>(llvm::divideCeil(k, tileCols) * tileCols);
    const bool needsPadding = paddedN != n || paddedK != k;

    OpBuilder builder(op);
    const Location loc = op.getLoc();
    Type i8 = builder.getI8Type();
    Type i32 = builder.getI32Type();
    Type actElem = actType.getElementType();
    Type outElem = outType.getElementType();
    Type weightElem = weightType.getElementType();
    (void)i8;
    (void)i32;

    // Device handle.
    auto deviceType = DeviceType::get(ctx, spec.name);
    Value device = builder.create<DeviceOpenOp>(loc, deviceType,
                                                 builder.getStringAttr(spec.name));

    // The weight sub-matrix a tile-block subview reads from, and the width
    // to slice the staged activation against -- the real weights/k in the
    // common case, or a zero-padded stand-in sized to the tile geometry
    // when N and/or K are ragged. Padding rows/columns are provably inert:
    // a row or column of zero weight can only ever contribute 0 * x to an
    // MVM's accumulator, so this cannot change any answer -- only the
    // write-back below (`validRows`) has to know which output rows are
    // real.
    Value workWeights = weights;
    int64_t workN = n;
    int64_t workK = k;
    if (needsPadding) {
      Value weightZero = builder.create<arith::ConstantOp>(
          loc, weightElem, builder.getIntegerAttr(weightElem, 0));
      auto paddedWeightType = MemRefType::get({paddedN, paddedK}, weightElem);
      Value paddedWeights =
          builder.create<memref::AllocOp>(loc, paddedWeightType);
      builder.create<linalg::FillOp>(loc, ValueRange{weightZero},
                                     ValueRange{paddedWeights});
      Value weightDst = subView(builder, loc, paddedWeights, {0, 0}, {n, k});
      builder.create<memref::CopyOp>(loc, weights, weightDst);
      workWeights = paddedWeights;
      workN = paddedN;
      workK = paddedK;
    }

    // K-padding scratch for the activation row, hoisted once regardless of
    // M: a plain memref (not a cim op), so cim-lower-to-target's automatic
    // per-loop buffer hoisting -- which only applies to buffers it and
    // cim ops allocate while lowering -- does not reach it. Allocating and
    // zero-filling it once, here, before any row's real data is copied
    // into its [0, k) prefix below, is what keeps M > 1 from leaking one
    // such buffer per row: the zero-padded tail [k, workK) never changes
    // across rows, so it only needs to be written once, ever.
    Value actPadded;
    if (workK != k) {
      Value actZero = builder.create<arith::ConstantOp>(
          loc, actElem, builder.getIntegerAttr(actElem, 0));
      auto paddedActType = MemRefType::get({workK}, actElem);
      actPadded = builder.create<memref::AllocOp>(loc, paddedActType);
      builder.create<linalg::FillOp>(loc, ValueRange{actZero},
                                     ValueRange{actPadded});
    }

    const int64_t tilesN = workN / tileRows;
    const int64_t tilesK = workK / tileCols;
    int64_t blockId = 0;

    auto tileType = TileType::get(ctx, {tileRows, tileCols}, actElem);
    auto residentType = ResidentType::get(ctx, {tileRows, tileCols}, actElem);
    auto partialType = nearMemRef(ctx, {tileRows}, outElem, SpaceKind::near);

    // Everything from here down is one row's worth of tiled matmul: stage
    // that row's activation, run the (compile-time-unrolled) N/K tile
    // blocks against it, write the row's result back. `rowOffset` is a
    // static zero for the m == 1 case -- textually identical IR to before
    // M > 1 support existed -- or the scf.for induction variable below.
    // cim.* ops here are emitted exactly once in program TEXT regardless
    // of M: for m == 1 this closure runs once, straight-line; for m > 1 it
    // runs once too, but its body sits inside a real runtime loop, so the
    // SAME emitted ops execute once per row (device buffer reuse across
    // those executions is cim-lower-to-target's job -- see the file
    // header).
    auto emitRow = [&](OpBuilder &b, OpFoldResult rowOffset) {
      Value actRow = rowSubView(b, loc, act, rowOffset, /*colOffset=*/0, k, {k});
      Value actSource = actRow;
      if (workK != k) {
        Value actDst = subView1D(b, loc, actPadded, 0, k);
        b.create<memref::CopyOp>(loc, actRow, actDst);
        actSource = actPadded;
      }
      auto actNearType = nearMemRef(ctx, {workK}, actElem, SpaceKind::near);
      Value actNear = b.create<CopyOp>(loc, actNearType, actSource);

      for (int64_t nb = 0; nb < tilesN; ++nb) {
        const int64_t n0 = nb * tileRows;
        SmallVector<Value> partials;

        for (int64_t kb = 0; kb < tilesK; ++kb) {
          const int64_t k0 = kb * tileCols;

          // The weight sub-matrix for this tile: W[n0:n0+rows, k0:k0+cols],
          // read from the padded stand-in when one was built above.
          Value weightBlock = subView(b, loc, workWeights, {n0, k0},
                                      {tileRows, tileCols});

          // Tile ids must stay within the target's declared tile count (spec
          // Sec. 5.4 rule 5). Handing out one id per block emits IR that asks
          // for tile 2 on a 2-tile device -- structurally valid, and rejected
          // by the runtime the moment it actually runs.
          //
          // Round-robin is safe *given this emission order*: each
          // cim.program is immediately followed by the cim.mvm that consumes
          // its resident, so a tile is never reprogrammed while a live
          // resident still refers to it. It is deliberately naive -- it
          // reprograms on every block even when the weights would still be
          // there. Choosing which weights stay resident is cim-placement's
          // job, and replacing this line is what wiring that pass up means.
          const int64_t tileId =
              spec.tiles.count > 0 ? blockId % spec.tiles.count : 0;
          Value tile = b.create<TileAllocOp>(loc, tileType, device,
                                              b.getI64IntegerAttr(tileId));

          // cim.program carries its own cost so later passes can reason about
          // reprogramming without a target lookup (spec Sec. 5.3).
          Value resident = b.create<ProgramOp>(
              loc, residentType, tile, weightBlock,
              b.getI64IntegerAttr(
                  static_cast<int64_t>(spec.costs.program.latencyNs)),
              b.getI64IntegerAttr(
                  static_cast<int64_t>(spec.costs.program.energyPj)),
              b.getBoolAttr(spec.tiles.persistent));

          Value actBlock = subView1D(b, loc, actNear, k0, tileCols);

          Value partial = b.create<MvmOp>(loc, partialType, resident,
                                          actBlock, b.getBoolAttr(false));
          partials.push_back(partial);
          ++blockId;
        }

        // Partial sums across the contraction dimension. With a single tile
        // in K there is nothing to reduce, and emitting a one-operand
        // reduction would just be noise for cim-placement to walk past.
        Value sum = partials.front();
        if (partials.size() > 1)
          sum = b.create<ReducePartialOp>(loc, partialType, partials);

        // Bring the result back to host memory and write it into the output
        // buffer. Requantization is deliberately not done here -- it is
        // cim-legalize-precision's job (spec Sec. 6, Pass 6).
        //
        // A block always produces tileRows results, but the *real* output
        // only has n rows -- when N was padded, the last block's tail rows
        // are the padding rows manufactured above, computed from an
        // all-zero weight sub-matrix and therefore always zero, but with no
        // home in `out` to be written to. validRows is tileRows for every
        // block except (at most) this last one, so this is a no-op change
        // in the exact-multiple case.
        const int64_t validRows = std::min(tileRows, n - n0);
        auto hostOutType = MemRefType::get({tileRows}, outElem);
        Value hostOut = b.create<CopyOp>(loc, hostOutType, sum);
        Value hostOutValid = validRows == tileRows
                                  ? hostOut
                                  : subView1D(b, loc, hostOut, 0, validRows);
        Value outSlice =
            rowSubView(b, loc, out, rowOffset, n0, validRows, {validRows});
        b.create<memref::CopyOp>(loc, hostOutValid, outSlice);
      }
    };

    if (m == 1) {
      emitRow(builder, builder.getI64IntegerAttr(0));
    } else {
      Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value cm = builder.create<arith::ConstantIndexOp>(loc, m);
      Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto forOp = builder.create<scf::ForOp>(loc, c0, cm, c1);
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(forOp.getBody());
      emitRow(bodyBuilder, forOp.getInductionVar());
    }

    LLVM_DEBUG(llvm::dbgs() << "cim-partition: lowered a " << m << "x" << n
                            << "x" << k << " matmul into " << blockId
                            << " tile block(s) per row\n");
    op->erase();
  }

  /// memref.subview with static offsets/sizes and unit strides.
  Value subView(OpBuilder &builder, Location loc, Value source,
                ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes) {
    auto sourceType = llvm::cast<MemRefType>(source.getType());
    SmallVector<int64_t> strides(sizes.size(), 1);
    auto resultType = llvm::cast<MemRefType>(memref::SubViewOp::inferResultType(
        sourceType, offsets, sizes, strides));
    return builder.create<memref::SubViewOp>(
        loc, resultType, source, toOpFold(builder, offsets),
        toOpFold(builder, sizes), toOpFold(builder, strides));
  }

  /// A rank-reduced 1D view of row `rowOffset` from a 2D memref, starting at
  /// column `colOffset` and `size` columns wide. `rowOffset` is an
  /// OpFoldResult rather than a plain int64_t specifically so a caller can
  /// pass a static zero (m == 1 -- identical IR to the pre-batching shape)
  /// or a dynamic scf.for induction variable (m > 1) through the same
  /// subview-construction logic.
  Value rowSubView(OpBuilder &builder, Location loc, Value source,
                   OpFoldResult rowOffset, int64_t colOffset, int64_t size,
                   ArrayRef<int64_t> resultShape) {
    auto sourceType = llvm::cast<MemRefType>(source.getType());
    SmallVector<OpFoldResult> offsets{rowOffset,
                                      builder.getI64IntegerAttr(colOffset)};
    SmallVector<OpFoldResult> sizes{builder.getI64IntegerAttr(1),
                                    builder.getI64IntegerAttr(size)};
    SmallVector<OpFoldResult> strides{builder.getI64IntegerAttr(1),
                                      builder.getI64IntegerAttr(1)};
    auto resultType = llvm::cast<MemRefType>(
        memref::SubViewOp::inferRankReducedResultType(resultShape, sourceType,
                                                       offsets, sizes, strides));
    return builder.create<memref::SubViewOp>(loc, resultType, source, offsets,
                                             sizes, strides);
  }

  Value subView1D(OpBuilder &builder, Location loc, Value source,
                  int64_t offset, int64_t size) {
    return subView(builder, loc, source, {offset}, {size});
  }

  SmallVector<OpFoldResult> toOpFold(OpBuilder &builder,
                                      ArrayRef<int64_t> values) {
    SmallVector<OpFoldResult> folded;
    folded.reserve(values.size());
    for (int64_t value : values)
      folded.push_back(builder.getI64IntegerAttr(value));
    return folded;
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMPartitionPass() {
  return std::make_unique<CIMPartitionPass>();
}
