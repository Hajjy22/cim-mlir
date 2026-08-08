// RUN: cim-opt %s --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file | FileCheck %s

// The case decision 5 in lib/Transforms/CIMPlacement.cpp's file header
// exists for: a weight repeated within one loop body, where the ordinary
// per-block Belady solve's own tile-0-first eviction tie-break stops
// finding the best available hoist. Hand-written cim ops rather than
// cim-detect/cim-partition output -- this exercises cim-placement in
// isolation against a use sequence engineered to hit exactly that
// tie-break, the same sequence test/unit/placement_test.cpp's own
// steady_state_handles_a_weight_repeated_within_one_body and
// steady_state_property_test.cpp's exhaustive coverage already prove
// correct at the engine level; this is the proof that the real pass
// actually reaches for it.
//
// One iteration's use sequence is [A, B, C, A, A, D] over 2 tiles -- A
// used three times (positions 0, 3, 4), B/C/D once each. Traced by hand
// (and in test/unit/placement_test.cpp): the ordinary per-block solve
// hoists NOTHING here (both tiles end up written twice within the block,
// so neither survives decision 3's own programCountPerTile == 1 check),
// leaving 4 cim.program ops per iteration. Deliberate pin-and-stream
// pins A (used most) and streams B, C, D through the remaining tile --
// 1 hoisted once, 3 per iteration -- strictly fewer over more than one
// iteration, so decision 5's cost comparison picks it.

memref.global "private" constant @wa : memref<4x4xi8> = dense<1>
memref.global "private" constant @wb : memref<4x4xi8> = dense<2>
memref.global "private" constant @wc : memref<4x4xi8> = dense<3>
memref.global "private" constant @wd : memref<4x4xi8> = dense<4>
memref.global "private" constant @act : memref<4xi8> = dense<5>

// CHECK-LABEL: func.func @repeated_weight_pins_deliberately
func.func @repeated_weight_pins_deliberately(%dev: !cim.device<"tiny-4x4">) {
  %wa = memref.get_global @wa : memref<4x4xi8>
  %wb = memref.get_global @wb : memref<4x4xi8>
  %wc = memref.get_global @wc : memref<4x4xi8>
  %wd = memref.get_global @wd : memref<4x4xi8>
  %act = memref.get_global @act : memref<4xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    // Step 0: A
    %ta0 = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"tiny-4x4">) -> !cim.tile<4x4xi8>
    %ra0 = cim.program %ta0, %wa {cost_ns = 1 : i64, cost_pj = 1 : i64}
           : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
    %oa0 = cim.mvm %ra0, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    memref.dealloc %oa0 : memref<4xi32>

    // Step 1: B
    %tb = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"tiny-4x4">) -> !cim.tile<4x4xi8>
    %rb = cim.program %tb, %wb {cost_ns = 1 : i64, cost_pj = 1 : i64}
          : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
    %ob = cim.mvm %rb, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    memref.dealloc %ob : memref<4xi32>

    // Step 2: C
    %tc = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"tiny-4x4">) -> !cim.tile<4x4xi8>
    %rc = cim.program %tc, %wc {cost_ns = 1 : i64, cost_pj = 1 : i64}
          : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
    %oc = cim.mvm %rc, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    memref.dealloc %oc : memref<4xi32>

    // Step 3: A again
    %ta1 = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"tiny-4x4">) -> !cim.tile<4x4xi8>
    %ra1 = cim.program %ta1, %wa {cost_ns = 1 : i64, cost_pj = 1 : i64}
           : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
    %oa1 = cim.mvm %ra1, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    memref.dealloc %oa1 : memref<4xi32>

    // Step 4: A a third time
    %ta2 = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"tiny-4x4">) -> !cim.tile<4x4xi8>
    %ra2 = cim.program %ta2, %wa {cost_ns = 1 : i64, cost_pj = 1 : i64}
           : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
    %oa2 = cim.mvm %ra2, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    memref.dealloc %oa2 : memref<4xi32>

    // Step 5: D
    %td = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"tiny-4x4">) -> !cim.tile<4x4xi8>
    %rd = cim.program %td, %wd {cost_ns = 1 : i64, cost_pj = 1 : i64}
          : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
    %od = cim.mvm %rd, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    memref.dealloc %od : memref<4xi32>
  }
  return
}

// Exactly one cim.program hoisted above the loop -- weight A, used three
// times, the one occurrence-count pinning picks -- not zero, which is
// what the ordinary per-block solve alone would have left here.
// CHECK: %[[RA:.*]] = cim.program %{{.*}}, %{{.*}} {{.*}} -> !cim.resident<4x4xi8>
// CHECK: scf.for
// The rewrite does not move ops -- each cim.program that survives (or
// each of A's own remaining mvm uses) stays exactly where it stood in
// the original textual order, which is why the three surviving
// cim.program ops (B, C, D) are interleaved with A's own two later
// mvm uses rather than grouped together. First: A's own first use,
// now consuming the hoisted resident directly (no cim.program for A
// survives inside the loop body at all).
// CHECK: cim.mvm %[[RA]],
// CHECK: cim.program {{.*}} -> !cim.resident<4x4xi8>
// CHECK: cim.program {{.*}} -> !cim.resident<4x4xi8>
// CHECK: cim.mvm %[[RA]],
// CHECK: cim.mvm %[[RA]],
// Exactly three cim.program ops total inside the loop (B, C, D above and
// D here) -- not four, which is what the ordinary per-block solve alone
// would emit.
// CHECK: cim.program {{.*}} -> !cim.resident<4x4xi8>
// CHECK-NOT: cim.program
