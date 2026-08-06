// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file | FileCheck %s
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file 2>&1 1>/dev/null | FileCheck --check-prefix=DIAG %s

// Cross-iteration reuse (decision 3 in CIMPlacement.cpp's header): a
// cim.program textually inside an scf.for body still costs a real reprogram
// on every runtime iteration unless something hoists it out. What is
// checked here is the *structure* of that hoist; test/mlir/pipeline_e2e_test.cpp
// checks that hoisting does not change a single computed value, and
// test/Run/placement-loop.mlir checks the saving reaches cimrt's own
// counters.

// One weight block, loop-invariant, used every iteration -- the tile fits
// the whole model, so the program must be hoisted entirely above the loop.

memref.global "private" constant @w : memref<4x4xi8> = dense<1>
memref.global "private" constant @acts : memref<3x4xi8> = dense<2>

func.func private @sink(memref<3x4xi32>)

// CHECK-LABEL: func.func @fits_hoists_entirely
func.func @fits_hoists_entirely() {
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
  func.call @sink(%out) : (memref<3x4xi32>) -> ()
  return
}

// The whole weight-side chain -- device_open, the weight subview, tile_alloc,
// program -- must appear before scf.for, in that dependency order, so each
// dominates its use once hoisted.
// CHECK: cim.device_open
// CHECK: memref.subview %{{.*}}[0, 0] [4, 4]
// CHECK: cim.tile_alloc
// CHECK: %[[R:.*]] = cim.program {{.*}} -> !cim.resident<4x4xi8>
// CHECK: scf.for
// The activation-side computation -- staging, the mvm, writing the output
// row -- stays inside, consuming the hoisted resident. No cim.program or
// cim.tile_alloc survives in the loop body: a reintroduced one anywhere
// between here and the loop's end would mean the hoist regressed.
// CHECK-NOT: cim.program
// CHECK-NOT: cim.tile_alloc
// CHECK: cim.mvm %[[R]],
// CHECK-NOT: cim.program
// CHECK-NOT: cim.tile_alloc
// CHECK: call @sink

// -----

// 12x4 weights over 4x4 tiles is 3 distinct blocks on a 2-tile device.
// Within one iteration the use sequence [0,1,2] has no repeats, so all 3
// programs survive per iteration regardless of placement -- but exactly one
// tile ends up holding its block for the whole iteration (stable, hoisted)
// while the other is reprogrammed mid-iteration (not stable, stays put).
// See the worked schedule in test/mlir/pipeline_e2e_test.cpp's
// placement_partially_hoists_when_only_some_tiles_are_stable.

memref.global "private" constant @w2 : memref<12x4xi8> = dense<1>
memref.global "private" constant @acts2 : memref<2x4xi8> = dense<2>

func.func private @sink2(memref<2x12xi32>)

// CHECK-LABEL: func.func @spill_hoists_only_the_stable_tile
func.func @spill_hoists_only_the_stable_tile() {
  %w = memref.get_global @w2 : memref<12x4xi8>
  %acts = memref.get_global @acts2 : memref<2x4xi8>
  %out = memref.alloc() : memref<2x12xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %actRow = memref.subview %acts[%i, 0] [1, 4] [1, 1]
      : memref<2x4xi8> to memref<1x4xi8, strided<[4, 1], offset: ?>>
    %actLocal = memref.alloc() : memref<1x4xi8>
    memref.copy %actRow, %actLocal
      : memref<1x4xi8, strided<[4, 1], offset: ?>> to memref<1x4xi8>
    %outRow = memref.subview %out[%i, 0] [1, 12] [1, 1]
      : memref<2x12xi32> to memref<1x12xi32, strided<[12, 1], offset: ?>>
    linalg.matmul_transpose_b ins(%actLocal, %w : memref<1x4xi8>, memref<12x4xi8>)
      outs(%outRow : memref<1x12xi32, strided<[12, 1], offset: ?>>)
    memref.dealloc %actLocal : memref<1x4xi8>
  }
  func.call @sink2(%out) : (memref<2x12xi32>) -> ()
  return
}

// Exactly one cim.program hoisted above the loop; exactly two remain inside
// it (one tile reprogrammed twice per iteration).
// CHECK: cim.program {{.*}} -> !cim.resident<4x4xi8>
// CHECK: scf.for
// CHECK-COUNT-2: cim.program {{.*}} -> !cim.resident<4x4xi8>
// CHECK: call @sink2

// -----

// A loop nested inside another scf.for is left entirely alone in v0.1: the
// inner loop's program must stay exactly where cim-partition put it, and a
// remark (checked in the DIAG run above) says why rather than silently
// doing nothing.

memref.global "private" constant @w3 : memref<4x4xi8> = dense<1>
memref.global "private" constant @acts3 : memref<2x4xi8> = dense<2>

func.func private @sink3(memref<2x4xi32>)

// CHECK-LABEL: func.func @nested_loop_is_left_alone
func.func @nested_loop_is_left_alone() {
  %w = memref.get_global @w3 : memref<4x4xi8>
  %acts = memref.get_global @acts3 : memref<2x4xi8>
  %out = memref.alloc() : memref<2x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %o = %c0 to %c2 step %c1 {
    scf.for %i = %c0 to %c2 step %c1 {
      %actRow = memref.subview %acts[%i, 0] [1, 4] [1, 1]
        : memref<2x4xi8> to memref<1x4xi8, strided<[4, 1], offset: ?>>
      %actLocal = memref.alloc() : memref<1x4xi8>
      memref.copy %actRow, %actLocal
        : memref<1x4xi8, strided<[4, 1], offset: ?>> to memref<1x4xi8>
      %outRow = memref.subview %out[%i, 0] [1, 4] [1, 1]
        : memref<2x4xi32> to memref<1x4xi32, strided<[4, 1], offset: ?>>
      linalg.matmul_transpose_b ins(%actLocal, %w : memref<1x4xi8>, memref<4x4xi8>)
        outs(%outRow : memref<1x4xi32, strided<[4, 1], offset: ?>>)
      memref.dealloc %actLocal : memref<1x4xi8>
    }
  }
  func.call @sink3(%out) : (memref<2x4xi32>) -> ()
  return
}

// CHECK: scf.for
// CHECK: scf.for
// CHECK: cim.program
// CHECK: call @sink3

// DIAG: not hoisting cim.program out of a loop nested inside another scf.for

// -----

// A loop whose trip count is not provably positive is left alone too: the
// program stays inside, and a remark explains why (checked in the DIAG run).

memref.global "private" constant @w4 : memref<4x4xi8> = dense<1>
memref.global "private" constant @acts4 : memref<1x4xi8> = dense<2>

func.func private @sink4(memref<1x4xi32>)

// CHECK-LABEL: func.func @dynamic_bound_is_left_alone
func.func @dynamic_bound_is_left_alone(%n: index) {
  %w = memref.get_global @w4 : memref<4x4xi8>
  %i0 = memref.get_global @acts4 : memref<1x4xi8>
  %a = memref.alloc() : memref<1x4xi8>
  memref.copy %i0, %a : memref<1x4xi8> to memref<1x4xi8>
  %out = memref.alloc() : memref<1x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<4x4xi8>)
      outs(%out : memref<1x4xi32>)
  }
  func.call @sink4(%out) : (memref<1x4xi32>) -> ()
  return
}

// CHECK: scf.for
// CHECK: cim.program
// CHECK: call @sink4

// DIAG: not hoisting cim.program out of this scf.for; its trip count is not provably positive

// -----

// lower == upper: the loop provably runs zero times. Hoisting a
// cim.program out would make it fire unconditionally even though the
// original IR never programs anything -- an energy/latency cost the
// source never asked for, even though no output value depends on it
// (nothing downstream of a loop that never runs can observe the
// difference in VALUES, which is exactly why this is a structural test,
// not a numerical one -- see the note on this in
// test/mlir/pipeline_e2e_test.cpp's
// placement_leaves_a_zero_trip_count_loop_alone).

memref.global "private" constant @w5 : memref<4x4xi8> = dense<1>
memref.global "private" constant @act5 : memref<1x4xi8> = dense<2>

func.func private @sink5(memref<1x4xi32>)

// CHECK-LABEL: func.func @zero_trip_count_is_left_alone
func.func @zero_trip_count_is_left_alone() {
  %w = memref.get_global @w5 : memref<4x4xi8>
  %i0 = memref.get_global @act5 : memref<1x4xi8>
  %a = memref.alloc() : memref<1x4xi8>
  memref.copy %i0, %a : memref<1x4xi8> to memref<1x4xi8>
  %out = memref.alloc() : memref<1x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c0 step %c1 {
    linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<4x4xi8>)
      outs(%out : memref<1x4xi32>)
  }
  func.call @sink5(%out) : (memref<1x4xi32>) -> ()
  return
}

// The program must still be INSIDE the loop -- CHECK, not a bare
// existence check, catches a program that moved above scf.for just as
// well as one that vanished.
// CHECK: scf.for
// CHECK: cim.program
// CHECK: call @sink5

// DIAG: not hoisting cim.program out of this scf.for; its trip count is not provably positive
