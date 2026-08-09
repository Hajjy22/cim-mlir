// RUN: cim-opt %s --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file --verify-diagnostics | FileCheck %s

// The capabilities.partial_sum_in_place: true sibling of
// cim-lower-to-target.mlir's reduce_two_partials_stays_device_until_copied/
// reduce_three_partials_chains_two_calls -- same op shapes, but tiny-4x4.yaml
// (unlike tiny-4x4-no-inplace.yaml) declares the capability, so
// lowerReducePartial takes the other branch: exactly ONE accumulator
// allocation for the whole chain (cimrt_copy from the first partial, then
// N-1 cimrt_reduce_add_inplace calls folding the rest into it), never one
// per chained step the way the general cimrt_reduce_add path needs.
// Verified against real cim-opt output before writing these CHECK lines,
// not hand-derived -- see lowerReducePartial's own comment in
// lib/Transforms/CIMLowerToTarget.cpp for why the first partial's own
// buffer is copied rather than mutated directly.

// CHECK-LABEL: func.func @reduce_two_partials_stays_device_until_copied
func.func @reduce_two_partials_stays_device_until_copied(
    %act0: memref<4xi8>, %act1: memref<4xi8>,
    %w0: memref<4x4xi8>, %w1: memref<4x4xi8>) -> memref<4xi32> {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %a0 = cim.copy %act0 : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %a1 = cim.copy %act1 : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %t0 = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %r0 = cim.program %t0, %w0 {cost_ns = 1 : i64, cost_pj = 1 : i64, persistent = true}
        : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  %p0 = cim.mvm %r0, %a0 {accumulate = false}
        : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>) -> memref<4xi32, #cim.space<near>>
  %t1 = cim.tile_alloc %dev {id = 1 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %r1 = cim.program %t1, %w1 {cost_ns = 1 : i64, cost_pj = 1 : i64, persistent = true}
        : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  // CHECK: call @cimrt_program
  // CHECK: call @cimrt_mvm
  %p1 = cim.mvm %r1, %a1 {accumulate = false}
        : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>) -> memref<4xi32, #cim.space<near>>
  // CHECK: call @cimrt_program
  // CHECK: call @cimrt_mvm
  %s = cim.reduce_partial %p0, %p1
       : (memref<4xi32, #cim.space<near>>, memref<4xi32, #cim.space<near>>) -> memref<4xi32, #cim.space<near>>
  // One accumulator, copied from the first mvm result (never mutated in
  // place -- that handle is not this pass' own scratch), then folded with
  // the second in place. No cimrt_reduce_add anywhere in this file.
  // CHECK: %[[ACC:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr
  // CHECK: call @cimrt_copy(%[[ACC]],
  // CHECK: call @cimrt_reduce_add_inplace(%{{[0-9]+}}, %[[ACC]],
  // CHECK-NOT: call @cimrt_reduce_add(
  // CHECK-NOT: call @cimrt_free(%[[ACC]])
  %h = cim.copy %s : memref<4xi32, #cim.space<near>> to memref<4xi32>
  // The reduction's own result stays a live device handle straight through
  // to this copy's read-back -- no extra alloc/free pair in between.
  // CHECK: call @cimrt_read(%[[ACC]],
  // CHECK-NOT: error
  return %h : memref<4xi32>
}

// -----

// Three operands: the headline claim. The general path (cim-lower-to-
// target.mlir's reduce_three_partials_chains_two_calls) needs TWO fresh
// accumulator allocations (N-1 for N=3) -- this target needs exactly ONE,
// regardless of operand count.
// CHECK-LABEL: func.func @reduce_three_partials_needs_only_one_accumulator
func.func @reduce_three_partials_needs_only_one_accumulator(
    %h0: memref<4xi32>, %h1: memref<4xi32>, %h2: memref<4xi32>) -> memref<4xi32> {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %p0 = cim.copy %h0 : memref<4xi32> to memref<4xi32, #cim.space<near>>
  %p1 = cim.copy %h1 : memref<4xi32> to memref<4xi32, #cim.space<near>>
  %p2 = cim.copy %h2 : memref<4xi32> to memref<4xi32, #cim.space<near>>
  // CHECK: call @cimrt_write(%[[P0BUF:[0-9]+]],
  // CHECK: call @cimrt_write(%[[P1BUF:[0-9]+]],
  // CHECK: call @cimrt_write(%[[P2BUF:[0-9]+]],
  %s = cim.reduce_partial %p0, %p1, %p2
       : (memref<4xi32, #cim.space<near>>, memref<4xi32, #cim.space<near>>,
          memref<4xi32, #cim.space<near>>) -> memref<4xi32>
  // Exactly one accumulator: copied from the first partial, then TWO
  // in-place folds -- never a second fresh "intermediate" buffer the
  // general path's own chain would need here.
  // CHECK: %[[ACC:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr
  // CHECK: call @cimrt_copy(%[[ACC]], %[[P0BUF]])
  // CHECK: call @cimrt_reduce_add_inplace(%{{[0-9]+}}, %[[ACC]], %[[P1BUF]],
  // CHECK: call @cimrt_reduce_add_inplace(%{{[0-9]+}}, %[[ACC]], %[[P2BUF]],
  // CHECK-NOT: call @cimrt_reduce_add(
  // A host-declared result: read back and freed, like the general path's
  // own host-result branch.
  // CHECK: call @cimrt_read(%[[ACC]],
  // CHECK: call @cimrt_free(%[[ACC]])
  // CHECK-NOT: error
  return %s : memref<4xi32>
}
