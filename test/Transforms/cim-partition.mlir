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

// cim.mvm is a matrix-vector primitive, so a multi-row (M > 1, "batching")
// matmul is lowered as the SAME per-row tile sequence as a single-row
// candidate, generated once and wrapped in a real scf.for over the M rows
// -- the exact IR shape test/real-target/check-loop.mlir.in already proves
// runs correctly as a real binary, here generated by this pass instead of
// hand-written. The device is opened once, OUTSIDE the loop (weights and
// the device handle do not depend on which row is being processed); the
// per-row activation stage/mvm/readback sequence is generated once but
// sits INSIDE the loop body, so it textually appears exactly once despite
// running M times.
// CHECK-LABEL: func.func @multi_row_is_batched_over_an_scf_for
func.func @multi_row_is_batched_over_an_scf_for(%act: memref<4x256xi8>, %out: memref<4x512xi32>) {
  %w = memref.get_global @weights_multi : memref<512x256xi8>
  // CHECK: cim.device_open
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]]

  // This row's activation, sliced with a DYNAMIC row offset (the
  // induction variable) rather than the static 0 every m == 1 candidate
  // uses elsewhere in this file.
  // CHECK: memref.subview %arg0[%[[IV]], 0] [1, 256] [1, 1]
  // CHECK: cim.copy

  // 512 / 256 tile rows = 2 N-blocks, 256 / 256 tile cols = 1 K-block per
  // block, so no cim.reduce_partial -- and each block's write-back also
  // uses the dynamic row offset.
  // CHECK: cim.tile_alloc {{.*}} {id = 0 : i64}
  // CHECK: cim.program
  // CHECK: cim.mvm
  // CHECK-NOT: cim.reduce_partial
  // CHECK: memref.subview %arg1[%[[IV]], 0] [1, 256] [1, 1]
  // CHECK: cim.tile_alloc {{.*}} {id = 1 : i64}
  // CHECK: cim.program
  // CHECK: cim.mvm
  // CHECK-NOT: cim.reduce_partial
  // CHECK: memref.subview %arg1[%[[IV]], 256] [1, 256] [1, 1]
  linalg.matmul_transpose_b ins(%act, %w : memref<4x256xi8>, memref<512x256xi8>)
                            outs(%out : memref<4x512xi32>)
  return
}
memref.global "private" constant @weights_multi : memref<512x256xi8> = dense<1>

// -----

// Weights in [K x N] layout do not match cim.mvm's output-major convention.
// Lowering them anyway would program transposed tiles and produce wrong
// numbers, so the candidate is left alone.
// A plain linalg.matmul (weights in ONNX's own [K, N] layout) used to be
// refused outright; it is now ACCEPTED, transposed into a fresh [N, K]
// buffer via a real scf.for this pass generates itself, the same
// treatment linalg.matmul_transpose_b's own weight already gets used
// as-is (docs/roadmap.md's M4 entry).
// CHECK-LABEL: func.func @plain_matmul_is_transposed_and_tiled
func.func @plain_matmul_is_transposed_and_tiled(%act: memref<1x256xi8>, %out: memref<1x512xi32>) {
  %w = memref.get_global @weights_kn : memref<256x512xi8>
  // The transpose: a fresh [N, K] = [512, 256] buffer, filled by a real
  // doubly-nested scf.for copying weights[k][n] into transposed[n][k].
  // CHECK: %[[TRANS:.*]] = memref.alloc() : memref<512x256xi8>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: memref.load
  // CHECK: memref.store %{{.*}}, %[[TRANS]]

  // Then tiled exactly like a matmul_transpose_b candidate would be:
  // 512 / 256 = 2 N-blocks against erbium-8t's 256x256 tiles, 256 / 256 =
  // 1 K-block each, so no cim.reduce_partial.
  // CHECK: cim.device_open
  // CHECK: cim.tile_alloc {{.*}} {id = 0 : i64}
  // CHECK: cim.program
  // CHECK: cim.mvm
  // CHECK-NOT: cim.reduce_partial
  // CHECK: cim.tile_alloc {{.*}} {id = 1 : i64}
  // CHECK: cim.program
  // CHECK: cim.mvm
  linalg.matmul ins(%act, %w : memref<1x256xi8>, memref<256x512xi8>)
                outs(%out : memref<1x512xi32>)
  return
}
memref.global "private" constant @weights_kn : memref<256x512xi8> = dense<1>
