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

// Ragged shapes must be refused rather than lowered to partially-filled
// tiles, which would compute silently wrong results. Spec Sec. 6 calls for
// zero-padding; until that exists the candidate stays as linalg.
// CHECK-LABEL: func.func @ragged_shape_is_left_alone
func.func @ragged_shape_is_left_alone(%act: memref<1x256xi8>, %out: memref<1x300xi32>) {
  %w = memref.get_global @weights_300x256 : memref<300x256xi8>
  // expected-warning @+2 {{not an exact multiple}}
  // CHECK: linalg.matmul_transpose_b
  linalg.matmul_transpose_b ins(%act, %w : memref<1x256xi8>, memref<300x256xi8>)
                            outs(%out : memref<1x300xi32>)
  // CHECK-NOT: cim.program
  return
}
memref.global "private" constant @weights_300x256 : memref<300x256xi8> = dense<1>

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
