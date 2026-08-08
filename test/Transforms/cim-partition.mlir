// RUN: cim-opt %s --cim-detect --cim-partition=target-yaml=%S/../../targets/erbium-8t.yaml --split-input-file --verify-diagnostics | FileCheck %s
//
// Pass 2 (spec Sec. 6): tile a matmul into per-tile cim.program/cim.mvm and
// reduce the partial sums. Tile geometry comes from the target file, never
// from a hardcoded constant -- that is the point of the target description.

// A 512x256 weight on erbium-8t's 256x256 tiles partitions into 2 blocks in
// N and 1 in K, so there is nothing to reduce.
// CHECK-LABEL: func.func @two_blocks_no_reduction
func.func @two_blocks_no_reduction(%act: memref<1x256xi8>, %out: memref<1x512xi32>) {
  %w = memref.get_global @weights_512x256 : memref<512x256xi8>

  // The device is opened once, and the activation staged into near memory
  // once -- not per tile (spec Sec. 3.4: transfers are explicit, so a
  // redundant one would show up in the cost report).
  // CHECK: cim.device_open
  // CHECK: cim.copy {{.*}} to memref<256xi8, #cim.space<near>>

  // CHECK: cim.tile_alloc {{.*}} {id = 0 : i64}
  // CHECK: cim.program {{.*}} {cost_ns = 12000 : i64, cost_pj = 480000 : i64}
  // CHECK: cim.mvm
  // CHECK: cim.tile_alloc {{.*}} {id = 1 : i64}
  // CHECK: cim.program
  // CHECK: cim.mvm

  // One tile in the contraction dimension means no partial sums to add.
  // CHECK-NOT: cim.reduce_partial

  // CHECK: return
  linalg.matmul_transpose_b ins(%act, %w : memref<1x256xi8>, memref<512x256xi8>)
                            outs(%out : memref<1x512xi32>)
  return
}
memref.global "private" constant @weights_512x256 : memref<512x256xi8> = dense<1>

// -----

// A 512x512 weight spans 2 blocks in N and 2 in K: four tiles, and each
// output block needs its two partial sums reduced.
// CHECK-LABEL: func.func @four_blocks_with_reduction
func.func @four_blocks_with_reduction(%act: memref<1x512xi8>, %out: memref<1x512xi32>) {
  %w = memref.get_global @weights_512x512 : memref<512x512xi8>
  // CHECK-COUNT-4: cim.program
  // CHECK-NOT: cim.program
  linalg.matmul_transpose_b ins(%act, %w : memref<1x512xi8>, memref<512x512xi8>)
                            outs(%out : memref<1x512xi32>)
  return
}
memref.global "private" constant @weights_512x512 : memref<512x512xi8> = dense<1>

// -----

// Ragged N: 300 rows is not a multiple of erbium-8t's 256-row tile, so the
// weight matrix is zero-padded up to 512 rows (linalg.fill, then the real
// 300x256 data copied into the top-left corner) before tiling proceeds
// exactly as in the exact-multiple case above. K is already exact, so no
// activation padding is needed.
// CHECK-LABEL: func.func @ragged_n_is_zero_padded
func.func @ragged_n_is_zero_padded(%act: memref<1x256xi8>, %out: memref<1x300xi32>) {
  %w = memref.get_global @weights_300x256 : memref<300x256xi8>

  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i8
  // CHECK: %[[PAD:.*]] = memref.alloc() : memref<512x256xi8>
  // CHECK: linalg.fill ins(%[[ZERO]] : i8) outs(%[[PAD]] : memref<512x256xi8>)
  // CHECK: %[[DST:.*]] = memref.subview %[[PAD]][0, 0] [300, 256] [1, 1]
  // CHECK: memref.copy %{{.*}}, %[[DST]]

  // Two N-blocks (512 / 256 tile rows), one K-block: nothing to reduce.
  // The first block is a full, real 256 rows, written back in between the
  // two cim.program ops...
  // CHECK: cim.program
  // CHECK: memref.copy %{{.*}} : memref<256xi32> to memref<256xi32, strided<[1]>>
  // CHECK: cim.program
  // CHECK-NOT: cim.program
  // CHECK-NOT: cim.reduce_partial

  // ...but the second block's tile computes 256 rows and only 44 of them
  // (300 - 256) are real output -- the rest are the zero-padded rows'
  // all-zero results, which have no home in %out.
  // CHECK: memref.subview %{{.*}}[0] [44] [1] : memref<256xi32> to memref<44xi32
  // CHECK: memref.subview %arg1[0, 256] [1, 44] [1, 1]
  linalg.matmul_transpose_b ins(%act, %w : memref<1x256xi8>, memref<300x256xi8>)
                            outs(%out : memref<1x300xi32>)
  return
}
memref.global "private" constant @weights_300x256 : memref<300x256xi8> = dense<1>

// -----

// Ragged K: 300 columns is not a multiple of erbium-8t's 256-column tile,
// so both the weight matrix and the staged activation are zero-padded up
// to 512 columns -- a padded activation column can only ever multiply a
// padded (zero) weight column, so it contributes 0 to every reduced sum
// and cannot change the answer. N is already exact, so the write-back is
// unaffected (see the N-ragged case above for that half).
// CHECK-LABEL: func.func @ragged_k_is_zero_padded
func.func @ragged_k_is_zero_padded(%act: memref<1x300xi8>, %out: memref<1x256xi32>) {
  %w = memref.get_global @weights_256x300 : memref<256x300xi8>

  // The weight matrix is padded first...
  // CHECK: %[[WZERO:.*]] = arith.constant 0 : i8
  // CHECK: %[[WPAD:.*]] = memref.alloc() : memref<256x512xi8>
  // CHECK: linalg.fill ins(%[[WZERO]] : i8) outs(%[[WPAD]] : memref<256x512xi8>)
  // CHECK: memref.subview %[[WPAD]][0, 0] [256, 300]

  // ...then the staged activation, separately, since it is a different
  // buffer with its own ragged edge to zero-pad up to the same 512.
  // CHECK: %[[AZERO:.*]] = arith.constant 0 : i8
  // CHECK: %[[APAD:.*]] = memref.alloc() : memref<512xi8>
  // CHECK: linalg.fill ins(%[[AZERO]] : i8) outs(%[[APAD]] : memref<512xi8>)
  // CHECK: memref.subview %[[APAD]][0] [300]

  // One N-block, two K-blocks (512 / 256 tile columns): the two partial
  // sums are reduced even though the second block is entirely padding --
  // this pass has no way to know a block is all-zero without inspecting
  // data it does not have at compile time, only shape.
  // CHECK-COUNT-2: cim.program
  // CHECK-NOT: cim.program
  // CHECK: cim.reduce_partial
  linalg.matmul_transpose_b ins(%act, %w : memref<1x300xi8>, memref<256x300xi8>)
                            outs(%out : memref<1x256xi32>)
  return
}
memref.global "private" constant @weights_256x300 : memref<256x300xi8> = dense<1>

// -----

// cim.mvm is a matrix-vector primitive; a multi-row matmul is outside the
// v0.1 contract and must not be silently lowered as if it were one row.
// CHECK-LABEL: func.func @multi_row_is_left_alone
func.func @multi_row_is_left_alone(%act: memref<4x256xi8>, %out: memref<4x512xi32>) {
  %w = memref.get_global @weights_multi : memref<512x256xi8>
  // expected-warning @+2 {{matrix-vector contract}}
  // CHECK: linalg.matmul_transpose_b
  linalg.matmul_transpose_b ins(%act, %w : memref<4x256xi8>, memref<512x256xi8>)
                            outs(%out : memref<4x512xi32>)
  // CHECK-NOT: cim.program
  return
}
memref.global "private" constant @weights_multi : memref<512x256xi8> = dense<1>

// -----

// Weights in [K x N] layout do not match cim.mvm's output-major convention.
// Lowering them anyway would program transposed tiles and produce wrong
// numbers, so the candidate is left alone.
// CHECK-LABEL: func.func @plain_matmul_layout_is_left_alone
func.func @plain_matmul_layout_is_left_alone(%act: memref<1x256xi8>, %out: memref<1x512xi32>) {
  %w = memref.get_global @weights_kn : memref<256x512xi8>
  // expected-warning @+2 {{output-major}}
  // CHECK: linalg.matmul
  linalg.matmul ins(%act, %w : memref<1x256xi8>, memref<256x512xi8>)
                outs(%out : memref<1x512xi32>)
  // CHECK-NOT: cim.program
  return
}
memref.global "private" constant @weights_kn : memref<256x512xi8> = dense<1>
