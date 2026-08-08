// The other half of cim-pipeline-full.mlir's composition claim, on real
// cim-partition output rather than hand-written IR:
// cim-pipeline-full.mlir's FULL run line proved all eight passes compose
// on a single-tile matmul; this file proves it on a genuine multi-K-tile
// one -- a shape the FULL chain could never take before this session,
// because cim-lower-to-target refused both cim.reduce_partial outright
// and the non-identity memref.subview cim-partition's subView1D emits
// slicing a staged activation once there is more than one K-tile (see
// CIMLowerToTarget.cpp's file header points 2 and its checkAllowedConsumers
// comment). Closing both is what makes this file's RUN line succeed at
// all: before, either one alone made cim-opt exit nonzero partway through
// this exact chain.
//
// A 4x8 weight on tiny-4x4.yaml's 4x4 tiles: 1 block in N, 2 in K, so
// cim-partition emits exactly the two-mvm, one-reduce_partial shape this
// session's two fixes exist for.
//
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   | FileCheck --check-prefix=MINIMAL %s
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-schedule \
// RUN:   --cim-insert-transfers \
// RUN:   --cim-legalize-precision=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   "--cim-cost-report=target-yaml=%S/../targets/tiny-4x4.yaml output=%t.json" \
// RUN:   --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   | FileCheck --check-prefix=FULL %s

func.func @two_k_blocks(%act: memref<1x8xi8>, %out: memref<1x4xi32>) {
  %w = memref.get_global @weights_4x8 : memref<4x8xi8>
  linalg.matmul_transpose_b ins(%act, %w : memref<1x8xi8>, memref<4x8xi8>)
                            outs(%out : memref<1x4xi32>)
  return
}
memref.global "private" constant @weights_4x8 : memref<4x8xi8> = dense<1>

// MINIMAL-LABEL: func.func @two_k_blocks
// The activation is staged once, whole (8 bytes)...
// MINIMAL: call @cimrt_write(%[[ACTBUF:[0-9]+]],
// ...then, per K-tile block (program first, matching cim-partition's own
// emission order -- tile_alloc/program, then the activation slice, then
// the mvm), sliced at that block's own byte offset (0, then 4) -- the
// non-identity subview materialization this session's second fix adds.
// MINIMAL: call @cimrt_program
// MINIMAL: call @cimrt_copy_range(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[ACTBUF]], %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}})
// MINIMAL: call @cimrt_mvm
// MINIMAL: call @cimrt_program
// MINIMAL: call @cimrt_copy_range(%{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}, %[[ACTBUF]], %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}})
// MINIMAL: call @cimrt_mvm
// Two partial sums reduced into one -- this session's first fix.
// MINIMAL: call @cimrt_reduce_add
// MINIMAL-NOT: call @cimrt_reduce_add
// MINIMAL-NOT: cim.device_open
// MINIMAL-NOT: cim.tile_alloc
// MINIMAL-NOT: cim.program
// MINIMAL-NOT: cim.mvm
// MINIMAL-NOT: cim.copy
// MINIMAL-NOT: cim.reduce_partial
// MINIMAL-NOT: memref.subview {{.*}}#cim.space
// MINIMAL-NOT: error

// FULL-LABEL: func.func @two_k_blocks
// The same claim, through literally all eight passes spec order -- the
// reduce_partial's result also picks up a real cim.requantize on the way
// (cim-legalize-precision), which must also survive this chain intact.
// FULL: call @cimrt_open
// FULL: call @cimrt_program
// FULL: call @cimrt_copy_range
// FULL: call @cimrt_mvm
// FULL: call @cimrt_program
// FULL: call @cimrt_copy_range
// FULL: call @cimrt_mvm
// FULL: call @cimrt_reduce_add
// FULL: call @cimrt_requantize
// FULL-NOT: cim.device_open
// FULL-NOT: cim.tile_alloc
// FULL-NOT: cim.program
// FULL-NOT: cim.mvm
// FULL-NOT: cim.copy
// FULL-NOT: cim.reduce_partial
// FULL-NOT: cim.requantize
// FULL-NOT: error
