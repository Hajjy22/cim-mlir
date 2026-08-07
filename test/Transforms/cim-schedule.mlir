// RUN: cim-opt %s --cim-schedule --split-input-file | FileCheck %s

// Pass 4 (spec Sec. 6). v0.1 keeps source order and inserts cim.barrier
// conservatively: nothing is reordered or overlapped (that is v0.2's
// double-buffering work, gated on the target's capabilities.double_buffer_program
// -- see docs/target-format.md), so the only question this pass answers is
// *where* a barrier must sit to guarantee every outstanding cim.program/
// cim.mvm on a device has completed before anything that might depend on it
// runs. cim.copy and cim.reduce_partial are exactly that "might depend on
// it": cim.copy commonly reads an mvm's own result, so a barrier appears
// before it rather than after -- inserting it after would let the copy
// race the mvm it is reading from on real (non-simulated) hardware.
//
// cim.barrier itself is a no-op in the functional simulator (it executes
// synchronously already), which is exactly why this file -- not a numerical
// differential -- is what actually catches a broken scheduler: running the
// output through cimrt would compute identical numbers whether or not a
// single barrier is missing. See test/mlir/pipeline_e2e_test.cpp for the
// "does not change the answer" half of this pass's contract, and
// test/Transforms/cim-schedule-pipeline.mlir for this pass running on IR
// cim-detect/cim-partition/cim-placement actually produce, not just
// hand-written cim ops.

// A single program+mvm run on one device: one barrier, placed after the
// run and before the function returns.
// CHECK-LABEL: func.func @straight_line_gets_one_barrier
func.func @straight_line_gets_one_barrier(%dev: !cim.device<"t">) {
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %act = memref.alloc() : memref<4xi8, #cim.space<near>>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
         -> memref<4xi32, #cim.space<near>>
  // CHECK: cim.program
  // CHECK: cim.mvm
  // CHECK: cim.barrier %{{.*}} : <"t">
  // CHECK-NOT: cim.barrier
  // CHECK: return
  return
}

// -----

// A cim.copy reading the mvm's own result sits between two runs of device
// work: the barrier must appear BEFORE the copy (so the copy cannot race
// the mvm it depends on), and a second barrier appears at the end for the
// second run.
// CHECK-LABEL: func.func @an_intervening_copy_forces_an_earlier_barrier
func.func @an_intervening_copy_forces_an_earlier_barrier(%dev: !cim.device<"t">) {
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %act = memref.alloc() : memref<4xi8, #cim.space<near>>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
         -> memref<4xi32, #cim.space<near>>
  %host = cim.copy %out : memref<4xi32, #cim.space<near>> to memref<4xi32>
  %out2 = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
          -> memref<4xi32, #cim.space<near>>
  // CHECK: cim.mvm
  // CHECK: cim.barrier %{{.*}} : <"t">
  // CHECK: cim.copy
  // CHECK: cim.mvm
  // CHECK: cim.barrier %{{.*}} : <"t">
  // CHECK-NOT: cim.barrier
  // CHECK: return
  return
}

// -----

// A resident hoisted above an scf.for (the loop-hoisting shape
// cim-placement produces, test/Transforms/cim-placement-loop.mlir) with an
// mvm inside the loop body. TWO barriers, both required:
//   1. Right before the loop -- the scf.for op's OWN operands are just its
//      bounds, never the resident (only something nested in its body
//      references that), so this is the case dependsOnAny's nested walk
//      exists for: without it, this barrier would be missed entirely and
//      the program's completion would not be guaranteed before the very
//      first iteration's mvm reads it.
//   2. Inside the loop body, right before scf.yield, so it fires once per
//      RUNTIME iteration -- guaranteeing iteration i's mvm has completed
//      before iteration i+1 issues its own, which the barrier before the
//      loop (fired only once, ever) cannot do.
// CHECK-LABEL: func.func @loop_body_gets_its_own_barrier
func.func @loop_body_gets_its_own_barrier(%dev: !cim.device<"t">) {
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  // CHECK: cim.program
  // CHECK: cim.barrier %{{.*}} : <"t">
  // CHECK: scf.for
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    %act = memref.alloc() : memref<4xi8, #cim.space<near>>
    %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
           -> memref<4xi32, #cim.space<near>>
    // CHECK: cim.mvm
    // CHECK-NEXT: cim.barrier %{{.*}} : <"t">
  }
  // Nothing pending after the loop -- both program (outside) and mvm
  // (inside) were already flushed at their own scope's exit.
  // CHECK-NOT: cim.barrier
  // CHECK: return
  return
}

// -----

// Two independent devices interleaved: each gets its own barrier at its own
// point, and neither device's work forces an early flush of the other's --
// there is nothing to synchronize between two unrelated devices.
// CHECK-LABEL: func.func @independent_devices_get_independent_barriers
func.func @independent_devices_get_independent_barriers(
    %devA: !cim.device<"a">, %devB: !cim.device<"b">) {
  %tA = cim.tile_alloc %devA {id = 0 : i64} : (!cim.device<"a">) -> !cim.tile<4x4xi8>
  %wA = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %rA = cim.program %tA, %wA {cost_ns = 1 : i64, cost_pj = 1 : i64}
        : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %tB = cim.tile_alloc %devB {id = 0 : i64} : (!cim.device<"b">) -> !cim.tile<4x4xi8>
  %wB = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %rB = cim.program %tB, %wB {cost_ns = 1 : i64, cost_pj = 1 : i64}
        : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %actA = memref.alloc() : memref<4xi8, #cim.space<near>>
  %outA = cim.mvm %rA, %actA : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
          -> memref<4xi32, #cim.space<near>>
  %actB = memref.alloc() : memref<4xi8, #cim.space<near>>
  %outB = cim.mvm %rB, %actB : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
          -> memref<4xi32, #cim.space<near>>
  // Both runs' work is emitted back-to-back with nothing non-device
  // interleaved, so both barriers land together at the end, one per device,
  // in the order their devices were first touched.
  // CHECK: cim.mvm
  // CHECK: cim.mvm
  // CHECK: cim.barrier %{{.*}} : <"a">
  // CHECK: cim.barrier %{{.*}} : <"b">
  // CHECK-NOT: cim.barrier
  // CHECK: return
  return
}
