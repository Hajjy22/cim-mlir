// RUN: cim-opt %s --cim-insert-transfers --split-input-file | FileCheck %s

// Pass 5 (spec Sec. 6). cim.mvm's activations operand must live in
// #cim.space<near> (spec Sec. 3.4). cim-partition's own output already
// stages every activation into near space itself before this pass would
// ever see it, so on today's real pipeline there is nothing for this pass
// to do -- the same documented gap as cim-legalize-precision's own
// isolation from cim-partition's output (see this pass's file header).
// Every case here is hand-written IR that manufactures the space mismatch
// cim-partition's current output never produces.

// A plain (unspaced, i.e. host) activation memref gets an explicit
// cim.copy to near space, and the mvm is rewired to read the copy's
// result rather than the original buffer.
// CHECK-LABEL: func.func @host_space_activation_gets_a_copy
func.func @host_space_activation_gets_a_copy(%r: !cim.resident<4x4xi8>) {
  %act = memref.alloc() : memref<4xi8>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
  memref.dealloc %out : memref<4xi32>
  // CHECK: %[[ACT:.*]] = memref.alloc()
  // CHECK-NEXT: %[[COPY:.*]] = cim.copy %[[ACT]] : memref<4xi8> to memref<4xi8, #cim.space<near>>
  // CHECK-NEXT: cim.mvm %{{.*}}, %[[COPY]]
  return
}

// -----

// An activation already in #cim.space<near> is left alone entirely -- no
// cim.copy is inserted. This is also what makes the pass idempotent:
// re-running it on its own output (a copy's near-space result) is a no-op.
// CHECK-LABEL: func.func @already_near_space_activation_is_left_alone
func.func @already_near_space_activation_is_left_alone(%r: !cim.resident<4x4xi8>) {
  %act = memref.alloc() : memref<4xi8, #cim.space<near>>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>) -> memref<4xi32>
  memref.dealloc %out : memref<4xi32>
  // CHECK-NOT: cim.copy
  // CHECK: cim.mvm
  return
}

// -----

// The activation is defined OUTSIDE the loop (loop-invariant), so the copy
// is inserted once, immediately before the scf.for, not once per
// iteration inside it.
// CHECK-LABEL: func.func @loop_invariant_activation_hoisted_above_the_loop
func.func @loop_invariant_activation_hoisted_above_the_loop(%r: !cim.resident<4x4xi8>) {
  %act = memref.alloc() : memref<4xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  // CHECK: cim.copy
  // CHECK-NEXT: scf.for
  scf.for %i = %c0 to %c3 step %c1 {
    %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    // CHECK: cim.mvm
    // CHECK-NOT: cim.copy
    memref.dealloc %out : memref<4xi32>
  }
  return
}

// -----

// The activation is computed INSIDE the loop body (a subview keyed off the
// induction variable), so it is loop-VARIANT and must not be hoisted: the
// copy stays inside the loop, immediately before the mvm.
// CHECK-LABEL: func.func @loop_variant_activation_copied_inside_the_loop
func.func @loop_variant_activation_copied_inside_the_loop(
    %r: !cim.resident<4x4xi8>, %acts: memref<3x4xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  // CHECK: scf.for
  scf.for %i = %c0 to %c3 step %c1 {
    %row = memref.subview %acts[%i, 0] [1, 4] [1, 1]
      : memref<3x4xi8> to memref<4xi8, strided<[1], offset: ?>>
    %out = cim.mvm %r, %row
      : (!cim.resident<4x4xi8>, memref<4xi8, strided<[1], offset: ?>>) -> memref<4xi32>
    // CHECK: memref.subview
    // CHECK-NEXT: cim.copy
    // CHECK-NEXT: cim.mvm
    memref.dealloc %out : memref<4xi32>
  }
  return
}

// -----

// Two mvms in the SAME loop share the exact same loop-invariant activation
// source: only ONE cim.copy is hoisted above the loop, and both mvms read
// it -- the "hoist redundant copies" half of Pass 5 is dedup, not merely
// hoisting.
// CHECK-LABEL: func.func @shared_invariant_activation_reused_by_two_mvms
func.func @shared_invariant_activation_reused_by_two_mvms(
    %r1: !cim.resident<4x4xi8>, %r2: !cim.resident<4x4xi8>) {
  %act = memref.alloc() : memref<4xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  // CHECK: %[[COPY:.*]] = cim.copy
  // CHECK-NEXT: scf.for
  scf.for %i = %c0 to %c3 step %c1 {
    %out1 = cim.mvm %r1, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    %out2 = cim.mvm %r2, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    // CHECK: cim.mvm %{{.*}}, %[[COPY]]
    // CHECK-NEXT: cim.mvm %{{.*}}, %[[COPY]]
    // CHECK-NOT: cim.copy
    memref.dealloc %out1 : memref<4xi32>
    memref.dealloc %out2 : memref<4xi32>
  }
  return
}
