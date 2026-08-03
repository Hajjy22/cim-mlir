//===- CIMPartition.cpp - Pass 2: tile a logical matmul --------*- C++ -*-===//

#include "PassDetail.h"
#include "cim/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::cim;

namespace {
struct CIMPartitionPass : public CIMPartitionBase<CIMPartitionPass> {
  void runOnOperation() override {
    // TODO(spec Sec.6, Pass 2): for each {cim.candidate} matmul of shape
    // [K x N], split into tiles of [tile_rows x tile_cols]:
    //   tiles_k = ceil(K / tile_rows)
    //   tiles_n = ceil(N / tile_cols)
    // Emit one cim.program + cim.mvm per sub-tile, plus a cim.reduce_partial
    // tree over the tiles_k dimension. Zero-pad ragged edges rather than
    // emitting special-cased tile shapes.
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMPartitionPass() {
  return std::make_unique<CIMPartitionPass>();
}
