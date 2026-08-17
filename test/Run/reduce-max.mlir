// RUN: cim-opt %s --cim-detect --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   | cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --profile - \
// RUN:   | FileCheck %s

// cim.reduce_max end to end through the shipped binaries, in the exact shape
// a MaxPool's kernel window uses: several STRIDED memref.subview taps of one
// padded buffer, folded elementwise by a single variadic op.
//
// THE OPERANDS ARE CHOSEN SO SIGNEDNESS IS VISIBLE. The two taps are
// [5, 3] and [-1, -128]:
//
//   signed max (correct, and what ONNX MaxPool on int8 computes) -> [5, 3]
//   unsigned byte compare (the copy-paste bug)                   -> [-1, -128]
//   an ADD, i.e. reduce_partial by mistake                       -> [4, -125]
//
// All three answers differ in BOTH elements, so this test cannot pass by
// luck under any of those three implementations. cimrt_reduce_max's own
// sign-extension is what makes the first one come out; test/unit/cimrt_test.cpp
// pins the same pair at the ABI level.
//
// The matmul is not incidental: cim-partition emits the cim.device_open that
// every cimrt_* call needs, and a pooling layer in a real network always
// follows a convolution, so this is also the realistic surrounding shape.

memref.global "private" constant @w : memref<2x2xi8> = dense<[[1, 0], [0, 1]]>
memref.global "private" constant @a : memref<1x2xi8> = dense<[[1, 2]]>
memref.global "private" constant @src : memref<1x4xi8> = dense<[[5, -1, 3, -128]]>
func.func private @cim_print_i8(memref<*xi8>)

func.func @main() {
  %w = memref.get_global @w : memref<2x2xi8>
  %aInit = memref.get_global @a : memref<1x2xi8>
  %a = memref.alloc() : memref<1x2xi8>
  memref.copy %aInit, %a : memref<1x2xi8> to memref<1x2xi8>
  %out = memref.alloc() : memref<1x2xi32>
  linalg.matmul_transpose_b ins(%a, %w : memref<1x2xi8>, memref<2x2xi8>)
    outs(%out : memref<1x2xi32>)

  %sInit = memref.get_global @src : memref<1x4xi8>
  %s = memref.alloc() : memref<1x4xi8>
  memref.copy %sInit, %s : memref<1x4xi8> to memref<1x4xi8>

  // Two taps of the same buffer, stride 2, offset 0 and 1 -- same SHAPE,
  // different layouts. cim.reduce_max's verifier constrains only the shape,
  // and the interpreter's gather walks arbitrary strides, so a pooling
  // window needs no per-tap copy into a contiguous slab.
  %tapA = memref.subview %s[0, 0] [1, 2] [1, 2]
    : memref<1x4xi8> to memref<1x2xi8, strided<[4, 2], offset: 0>>
  %tapB = memref.subview %s[0, 1] [1, 2] [1, 2]
    : memref<1x4xi8> to memref<1x2xi8, strided<[4, 2], offset: 1>>
  %r = cim.reduce_max %tapA, %tapB
    : (memref<1x2xi8, strided<[4, 2], offset: 0>>,
       memref<1x2xi8, strided<[4, 2], offset: 1>>) -> memref<1x2xi8>

  %u = memref.cast %r : memref<1x2xi8> to memref<*xi8>
  func.call @cim_print_i8(%u) : (memref<*xi8>) -> ()
  memref.dealloc %a : memref<1x2xi8>
  memref.dealloc %out : memref<1x2xi32>
  memref.dealloc %s : memref<1x4xi8>
  return
}

// CHECK: cim_print_i8 shape=[1,2] data=[5,3]

// Charged, not free: two operands fold to N-1 = 1 cimrt_reduce_max call,
// counted separately from the adder (which never fires here). This is the
// runtime half of the static-vs-runtime invariant cimrt.h records -- every
// op the runtime can execute is charged against the target's cost table.
// CHECK: cimrt_profile {{.*}}reduce_adds=0 reduce_maxes=1
