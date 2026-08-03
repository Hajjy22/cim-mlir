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
//   - a single output row: the v0.1 contract is matrix-vector
//   - N and K exact multiples of the tile geometry. Spec Sec. 6 calls for
//     zero-padding ragged edges; that needs a pad-and-copy sequence and is
//     not implemented yet.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Target/TargetSpec.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

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

    if (m != 1) {
      op->emitWarning("cim-partition implements the v0.1 matrix-vector "
                      "contract (a single output row); this matmul has ")
          << m << " rows and is left unoffloaded";
      return;
    }
    if (weightType.getShape()[1] != k || outType.getShape()[1] != n) {
      op->emitWarning("cim-partition: inconsistent matmul shapes; leaving this "
                      "candidate unoffloaded");
      return;
    }
    if (n % tileRows != 0 || k % tileCols != 0) {
      // Spec Sec. 6 says zero-pad the ragged edge. Emitting a partial tile
      // instead would compute silently wrong results, so refuse.
      op->emitWarning("cim-partition: weight shape ")
          << n << "x" << k << " is not an exact multiple of the target's "
          << tileRows << "x" << tileCols
          << " tile geometry; zero-padding of ragged edges is not implemented "
             "yet, so this candidate is left unoffloaded";
      return;
    }

    OpBuilder builder(op);
    const Location loc = op.getLoc();
    Type i8 = builder.getI8Type();
    Type i32 = builder.getI32Type();
    Type actElem = actType.getElementType();
    Type outElem = outType.getElementType();
    (void)i8;
    (void)i32;

    // Device handle.
    auto deviceType = DeviceType::get(ctx, spec.name);
    Value device = builder.create<DeviceOpenOp>(loc, deviceType,
                                                 builder.getStringAttr(spec.name));

    // Activations: collapse the 1 x K input to a K-vector, then stage it in
    // near memory once. Every tile reads a slice of that one staged copy --
    // spec Sec. 3.4 requires the transfer to be explicit, and hoisting it
    // out of the tile loop is the whole point of making it visible.
    Value actRow = rankReducedSubView(builder, loc, act, {0, 0}, {1, k}, {k});
    auto actNearType = nearMemRef(ctx, {k}, actElem, SpaceKind::near);
    Value actNear = builder.create<CopyOp>(loc, actNearType, actRow);

    const int64_t tilesN = n / tileRows;
    const int64_t tilesK = k / tileCols;
    int64_t blockId = 0;

    auto tileType = TileType::get(ctx, {tileRows, tileCols}, actElem);
    auto residentType = ResidentType::get(ctx, {tileRows, tileCols}, actElem);
    auto partialType = nearMemRef(ctx, {tileRows}, outElem, SpaceKind::near);

    for (int64_t nb = 0; nb < tilesN; ++nb) {
      const int64_t n0 = nb * tileRows;
      SmallVector<Value> partials;

      for (int64_t kb = 0; kb < tilesK; ++kb) {
        const int64_t k0 = kb * tileCols;

        // The weight sub-matrix for this tile: W[n0:n0+rows, k0:k0+cols].
        Value weightBlock = subView(builder, loc, weights, {n0, k0},
                                    {tileRows, tileCols});

        Value tile = builder.create<TileAllocOp>(loc, tileType, device,
                                                  builder.getI64IntegerAttr(blockId));

        // cim.program carries its own cost so later passes can reason about
        // reprogramming without a target lookup (spec Sec. 5.3).
        Value resident = builder.create<ProgramOp>(
            loc, residentType, tile, weightBlock,
            builder.getI64IntegerAttr(
                static_cast<int64_t>(spec.costs.program.latencyNs)),
            builder.getI64IntegerAttr(
                static_cast<int64_t>(spec.costs.program.energyPj)),
            builder.getBoolAttr(spec.tiles.persistent));

        Value actBlock = subView1D(builder, loc, actNear, k0, tileCols);

        Value partial = builder.create<MvmOp>(loc, partialType, resident,
                                               actBlock, builder.getBoolAttr(false));
        partials.push_back(partial);
        ++blockId;
      }

      // Partial sums across the contraction dimension. With a single tile in
      // K there is nothing to reduce, and emitting a one-operand reduction
      // would just be noise for cim-placement to walk past.
      Value sum = partials.front();
      if (partials.size() > 1)
        sum = builder.create<ReducePartialOp>(loc, partialType, partials);

      // Bring the result back to host memory and write it into the output
      // buffer. Requantization is deliberately not done here -- it is
      // cim-legalize-precision's job (spec Sec. 6, Pass 6).
      auto hostOutType = MemRefType::get({tileRows}, outElem);
      Value hostOut = builder.create<CopyOp>(loc, hostOutType, sum);
      Value outSlice =
          rankReducedSubView(builder, loc, out, {0, n0}, {1, tileRows}, {tileRows});
      builder.create<memref::CopyOp>(loc, hostOut, outSlice);
    }

    LLVM_DEBUG(llvm::dbgs() << "cim-partition: lowered a " << n << "x" << k
                            << " matmul into " << blockId << " tile block(s)\n");
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

  /// memref.subview that also drops unit dimensions.
  Value rankReducedSubView(OpBuilder &builder, Location loc, Value source,
                           ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes,
                           ArrayRef<int64_t> resultShape) {
    auto sourceType = llvm::cast<MemRefType>(source.getType());
    SmallVector<int64_t> strides(sizes.size(), 1);
    auto resultType = llvm::cast<MemRefType>(
        memref::SubViewOp::inferRankReducedResultType(resultShape, sourceType,
                                                       offsets, sizes, strides));
    return builder.create<memref::SubViewOp>(
        loc, resultType, source, toOpFold(builder, offsets),
        toOpFold(builder, sizes), toOpFold(builder, strides));
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
