// RUN: cim-opt %s --cim-detect --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   | cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --profile - \
// RUN:   | FileCheck %s

// The shell-driven counterpart to test/mlir/pipeline_e2e_test.cpp. That test
// runs the passes and the interpreter in one process, which is what makes it
// usable under ASan and gcov; this one drives the shipped binaries over
// textual IR, so it also covers the printer/parser round trip between the two
// and the command-line plumbing neither of them shares.

memref.global "private" constant @w : memref<8x8xi8> = dense<[
  [ 1,  0,  0,  0,  0,  0,  0,  0],
  [ 0,  1,  0,  0,  0,  0,  0,  0],
  [ 0,  0,  1,  0,  0,  0,  0,  0],
  [ 0,  0,  0,  1,  0,  0,  0,  0],
  [ 1,  1,  1,  1,  1,  1,  1,  1],
  [ 1, -1,  1, -1,  1, -1,  1, -1],
  [ 2,  2,  2,  2, -2, -2, -2, -2],
  [ 0,  0,  0,  0,  0,  0,  0, 100]]>

memref.global "private" constant @a : memref<1x8xi8> = dense<[
  [10, 20, 30, 40, -1, -2, -3, -4]]>

func.func private @cim_print_i32(memref<*xi32>)

func.func @main() {
  %w = memref.get_global @w : memref<8x8xi8>
  %aInit = memref.get_global @a : memref<1x8xi8>
  %a = memref.alloc() : memref<1x8xi8>
  memref.copy %aInit, %a : memref<1x8xi8> to memref<1x8xi8>
  %out = memref.alloc() : memref<1x8xi32>
  linalg.matmul_transpose_b
    ins(%a, %w : memref<1x8xi8>, memref<8x8xi8>)
    outs(%out : memref<1x8xi32>)
  %u = memref.cast %out : memref<1x8xi32> to memref<*xi32>
  func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()
  memref.dealloc %a : memref<1x8xi8>
  memref.dealloc %out : memref<1x8xi32>
  return
}

// Rows 0-3 select one activation each; row 4 sums them all (10+20+30+40-1-2-3-4
// = 90); row 5 alternates signs (10-20+30-40-1+2-3+4 = -18); row 6 is
// 2*(10+20+30+40) - 2*(-1-2-3-4) = 200 + 20 = 220; row 7 is 100 * -4.
// CHECK: cim_print_i32 shape=[1,8] data=[10,20,30,40,90,-18,220,-400]

// 8x8 over 4x4 tiles is 4 blocks, and the target has 2 tiles -- so the
// counters must show reprogramming, not one big offload. Without this the
// numbers above would still pass if cim-partition stopped tiling.
// CHECK: cimrt_profile programs=4 mvms=4
