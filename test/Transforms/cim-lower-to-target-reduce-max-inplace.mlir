// RUN: cim-opt %s --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4-max-inplace.yaml \
// RUN:   --split-input-file --verify-diagnostics | FileCheck %s

// The capabilities.max_in_place: true sibling of cim-lower-to-target.mlir's
// reduce_two_taps_folds_one_max/reduce_three_taps_chains_two_max_calls --
// same op shapes, but tiny-4x4-max-inplace.yaml (unlike
// tiny-4x4-no-inplace.yaml) declares the capability, so lowerReduceMax
// takes its other branch: exactly ONE accumulator allocation for the whole
// chain (cimrt_copy from the first operand, then N-1
// cimrt_reduce_max_inplace calls folding the rest into it), never one per
// chained step the way the general cimrt_reduce_max path needs. Mirrors
// cim-lower-to-target-reduce-partial-inplace.mlir exactly, with
// cimrt_reduce_max/cimrt_reduce_max_inplace substituted for
// cimrt_reduce_add/cimrt_reduce_add_inplace.
//
// Verified against real cim-opt output before writing these CHECK lines,
// not hand-derived -- see lowerReduceMax's own comment in
// lib/Transforms/CIMLowerToTarget.cpp for why the first operand's own
// staged buffer is copied rather than mutated directly.

// CHECK-LABEL: func.func @reduce_two_taps_folds_in_place
func.func @reduce_two_taps_folds_in_place(
    %t0: memref<4xi8>, %t1: memref<4xi8>) -> memref<4xi8> {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %p0 = cim.copy %t0 : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %p1 = cim.copy %t1 : memref<4xi8> to memref<4xi8, #cim.space<near>>
  // CHECK: call @cimrt_write(%[[P0BUF:[0-9]+]],
  // CHECK: call @cimrt_write(%[[P1BUF:[0-9]+]],
  %m = cim.reduce_max %p0, %p1
       : (memref<4xi8, #cim.space<near>>, memref<4xi8, #cim.space<near>>) -> memref<4xi8, #cim.space<near>>
  // One accumulator, copied from the first operand (never mutated in
  // place -- that handle is not this pass' own scratch), then folded with
  // the second in place. No cimrt_reduce_max anywhere in this file.
  // CHECK: %[[ACC:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr
  // CHECK: call @cimrt_copy(%[[ACC]], %[[P0BUF]])
  // CHECK: call @cimrt_reduce_max_inplace(%{{[0-9]+}}, %[[ACC]], %[[P1BUF]],
  // CHECK-NOT: call @cimrt_reduce_max(
  // CHECK-NOT: call @cimrt_free(%[[ACC]])
  %h = cim.copy %m : memref<4xi8, #cim.space<near>> to memref<4xi8>
  // CHECK: call @cimrt_read(%[[ACC]],
  // CHECK-NOT: error
  return %h : memref<4xi8>
}

// -----

// Three operands: the general path (cim-lower-to-target.mlir's
// reduce_three_taps_chains_two_max_calls) needs TWO fresh accumulator
// allocations (N-1 for N=3) -- this target needs exactly ONE, regardless
// of operand count.
// CHECK-LABEL: func.func @reduce_three_taps_needs_only_one_accumulator
func.func @reduce_three_taps_needs_only_one_accumulator(
    %h0: memref<4xi8>, %h1: memref<4xi8>, %h2: memref<4xi8>) -> memref<4xi8> {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %p0 = cim.copy %h0 : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %p1 = cim.copy %h1 : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %p2 = cim.copy %h2 : memref<4xi8> to memref<4xi8, #cim.space<near>>
  // CHECK: call @cimrt_write(%[[P0BUF:[0-9]+]],
  // CHECK: call @cimrt_write(%[[P1BUF:[0-9]+]],
  // CHECK: call @cimrt_write(%[[P2BUF:[0-9]+]],
  %m = cim.reduce_max %p0, %p1, %p2
       : (memref<4xi8, #cim.space<near>>, memref<4xi8, #cim.space<near>>,
          memref<4xi8, #cim.space<near>>) -> memref<4xi8>
  // Exactly one accumulator: copied from the first operand, then TWO
  // in-place folds -- never a second fresh "intermediate" buffer the
  // general path's own chain would need here.
  // CHECK: %[[ACC:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr
  // CHECK: call @cimrt_copy(%[[ACC]], %[[P0BUF]])
  // CHECK: call @cimrt_reduce_max_inplace(%{{[0-9]+}}, %[[ACC]], %[[P1BUF]],
  // CHECK: call @cimrt_reduce_max_inplace(%{{[0-9]+}}, %[[ACC]], %[[P2BUF]],
  // CHECK-NOT: call @cimrt_reduce_max(
  // A host-declared result: read back and freed, like the general path's
  // own host-result branch.
  // CHECK: call @cimrt_read(%[[ACC]],
  // CHECK: call @cimrt_free(%[[ACC]])
  // CHECK-NOT: error
  return %m : memref<4xi8>
}
