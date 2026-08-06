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

// The same module, executed with and without cim-placement, checked from a
// shell. test/mlir/pipeline_e2e_test.cpp already proves the values match
// in-process; what this adds is that the saving reaches the *runtime*.
// Eliminating a cim.program in the IR is only worth something if it
// eliminates a cimrt_program call, and the profile counters are the only
// place that is visible.

memref.global "private" constant @w : memref<8x4xi8> = dense<[
  [ 1,  0,  0,  0],
  [ 0,  1,  0,  0],
  [ 0,  0,  1,  0],
  [ 0,  0,  0,  1],
  [ 1,  1,  1,  1],
  [ 2,  0,  0,  0],
  [ 0,  2,  0,  0],
  [ 1, -1,  1, -1]]>

memref.global "private" constant @a1 : memref<1x4xi8> = dense<[[10, 20, 30, 40]]>
memref.global "private" constant @a2 : memref<1x4xi8> = dense<[[1, 2, 3, 4]]>

func.func private @cim_print_i32(memref<*xi32>)

func.func @main() {
  %w = memref.get_global @w : memref<8x4xi8>

  %i1 = memref.get_global @a1 : memref<1x4xi8>
  %a1 = memref.alloc() : memref<1x4xi8>
  memref.copy %i1, %a1 : memref<1x4xi8> to memref<1x4xi8>
  %o1 = memref.alloc() : memref<1x8xi32>
  linalg.matmul_transpose_b ins(%a1, %w : memref<1x4xi8>, memref<8x4xi8>)
    outs(%o1 : memref<1x8xi32>)
  %u1 = memref.cast %o1 : memref<1x8xi32> to memref<*xi32>
  func.call @cim_print_i32(%u1) : (memref<*xi32>) -> ()

  %i2 = memref.get_global @a2 : memref<1x4xi8>
  %a2 = memref.alloc() : memref<1x4xi8>
  memref.copy %i2, %a2 : memref<1x4xi8> to memref<1x4xi8>
  %o2 = memref.alloc() : memref<1x8xi32>
  linalg.matmul_transpose_b ins(%a2, %w : memref<1x4xi8>, memref<8x4xi8>)
    outs(%o2 : memref<1x8xi32>)
  %u2 = memref.cast %o2 : memref<1x8xi32> to memref<*xi32>
  func.call @cim_print_i32(%u2) : (memref<*xi32>) -> ()

  memref.dealloc %a1 : memref<1x4xi8>
  memref.dealloc %o1 : memref<1x8xi32>
  memref.dealloc %a2 : memref<1x4xi8>
  memref.dealloc %o2 : memref<1x8xi32>
  return
}

// Rows 0-3 select one activation each; row 4 sums them; row 5 and 6 double
// one; row 7 alternates signs. Both runs must print exactly this.
// PLAIN:  cim_print_i32 shape=[1,8] data=[10,20,30,40,100,20,40,-20]
// PLAIN:  cim_print_i32 shape=[1,8] data=[1,2,3,4,10,2,4,-2]
// PLACED: cim_print_i32 shape=[1,8] data=[10,20,30,40,100,20,40,-20]
// PLACED: cim_print_i32 shape=[1,8] data=[1,2,3,4,10,2,4,-2]

// 8x4 weights over 4x4 tiles is 2 blocks, and the target has 2 tiles, so
// everything fits: the second matmul reprograms nothing. mvms must NOT
// change -- placement removes redundant weight writes, not computation, and
// a pass that dropped an mvm would still print the values above for the
// first matmul.
// PLAIN:  cimrt_profile programs=4 mvms=4
// PLACED: cimrt_profile programs=2 mvms=4
