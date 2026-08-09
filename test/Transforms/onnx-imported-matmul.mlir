// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   | FileCheck %s

// This module was PRODUCED BY the ONNX front end (python/cim_frontend),
// from an 8x4 MatMulInteger model, and is checked in verbatim.
//
// It exists so the `mlir` CI job -- which installs no ONNX packages and
// so skips test/python/test_onnx_frontend.py entirely -- still fails if
// the shape the importer emits stops compiling. Without it, a change that
// broke the front end's output would be caught only in the one job that
// has the optional dependencies installed.
//
// test/python/test_onnx_frontend.py::test_checked_in_lit_fixture_is_current
// regenerates this file and diffs it, so it cannot silently go stale.
//
// Note the weight literal is the ONNX weight TRANSPOSED: ONNX stores B as
// [K, N] and cim.mvm indexes W[n][k], so the importer flips it. The model
// this came from had B[0] = [1, -2, 3, -4, 5, -6, 7, -8]; here that is
// the first COLUMN.

memref.global "private" constant @W : memref<8x4xi8> = dense<[[1, 9, 17, 25], [-2, -10, -18, -26], [3, 11, 19, 27], [-4, -12, -20, -28], [5, 13, 21, 29], [-6, -14, -22, -30], [7, 15, 23, 31], [-8, -16, -24, -32]]>
memref.global "private" constant @A : memref<1x4xi8> = dense<[[2, -3, 5, -7]]>
func.func private @cim_print_i32(memref<*xi32>)
func.func @main() {
  %w = memref.get_global @W : memref<8x4xi8>
  %aInit = memref.get_global @A : memref<1x4xi8>
  %a = memref.alloc() : memref<1x4xi8>
  memref.copy %aInit, %a : memref<1x4xi8> to memref<1x4xi8>
  %out = memref.alloc() : memref<1x8xi32>
  linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<8x4xi8>)
    outs(%out : memref<1x8xi32>)
  %u = memref.cast %out : memref<1x8xi32> to memref<*xi32>
  func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()
  memref.dealloc %a : memref<1x4xi8>
  memref.dealloc %out : memref<1x8xi32>
  return
}

// The whole point: the imported module reaches cim ops. An importer that
// emitted a subtly different shape -- most plausibly by dropping the
// memref.alloc/memref.copy staging of the activation, which cim-detect
// needs in order to see exactly one constant operand -- would leave the
// linalg op alone and emit no cim.program at all.
// CHECK-LABEL: func.func @main
// CHECK: cim.device_open
// CHECK: cim.tile_alloc
// CHECK: cim.program
// CHECK: cim.mvm
// CHECK-NOT: linalg.matmul_transpose_b
