// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   | cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --profile - \
// RUN:   | FileCheck --check-prefix=PLAIN %s
//
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   | cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --profile - \
// RUN:   | FileCheck --check-prefix=PLACED %s

// test/Transforms/cim-placement-loop.mlir checks that the hoisted IR has the
// right shape; test/mlir/pipeline_e2e_test.cpp checks hoisting never changes
// a computed value. What neither of those runs is the loop for real: this
// drives cim-run over a genuine 3-iteration scf.for and reads cimrt's own
// counters, which is the only place "eliminated a redundant reprogram"
// means anything a real device would notice.
//
// Without cim-placement, the single textual cim.program still fires once
// per RUNTIME iteration -- three programs for three iterations, even though
// there is exactly one cim.program in the IR either way. With placement,
// the same weight block is proven loop-invariant and hoisted above the
// loop: one program total, however many iterations run.

memref.global "private" constant @w : memref<4x4xi8> = dense<[
  [ 1,  0,  0,  0],
  [ 0,  1,  0,  0],
  [ 0,  0,  1,  0],
  [ 0,  0,  0,  1]]>

memref.global "private" constant @acts : memref<3x4xi8> = dense<[
  [ 1,  2,  3,  4],
  [10, 20, 30, 40],
  [-1, -2, -3, -4]]>

func.func private @cim_print_i32(memref<*xi32>)

func.func @main() {
  %w = memref.get_global @w : memref<4x4xi8>
  %acts = memref.get_global @acts : memref<3x4xi8>
  %out = memref.alloc() : memref<3x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    %actRow = memref.subview %acts[%i, 0] [1, 4] [1, 1]
      : memref<3x4xi8> to memref<1x4xi8, strided<[4, 1], offset: ?>>
    %actLocal = memref.alloc() : memref<1x4xi8>
    memref.copy %actRow, %actLocal
      : memref<1x4xi8, strided<[4, 1], offset: ?>> to memref<1x4xi8>
    %outRow = memref.subview %out[%i, 0] [1, 4] [1, 1]
      : memref<3x4xi32> to memref<1x4xi32, strided<[4, 1], offset: ?>>
    linalg.matmul_transpose_b ins(%actLocal, %w : memref<1x4xi8>, memref<4x4xi8>)
      outs(%outRow : memref<1x4xi32, strided<[4, 1], offset: ?>>)
    memref.dealloc %actLocal : memref<1x4xi8>
  }
  %cast = memref.cast %out : memref<3x4xi32> to memref<*xi32>
  func.call @cim_print_i32(%cast) : (memref<*xi32>) -> ()
  memref.dealloc %out : memref<3x4xi32>
  return
}

// Identity weights: each row of the output equals the matching activation
// row. Both runs must print exactly this -- hoisting must not touch a
// single value.
// PLAIN:  cim_print_i32 shape=[3,4] data=[1,2,3,4,10,20,30,40,-1,-2,-3,-4]
// PLACED: cim_print_i32 shape=[3,4] data=[1,2,3,4,10,20,30,40,-1,-2,-3,-4]

// The counters are what this file exists for: mvms stays at 3 (one
// computation per iteration, unchanged by placement), and programs drops
// from 3 (one cimrt_program call per runtime iteration) to 1 (hoisted:
// programmed once, reused for the other two iterations).
// PLAIN:  cimrt_profile programs=3 mvms=3
// PLACED: cimrt_profile programs=1 mvms=3
