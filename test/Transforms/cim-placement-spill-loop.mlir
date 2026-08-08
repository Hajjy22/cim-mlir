// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4-8t.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4-8t.yaml \
// RUN:   | FileCheck %s
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4-8t.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4-8t.yaml \
// RUN:   --cim-cost-report=target-yaml=%S/../targets/tiny-4x4-8t.yaml \
// RUN:   | FileCheck --check-prefix=COST %s

// Spill under a loop, on a target with enough tiles for spill to be a real
// question (tiny-4x4.yaml's 2 tiles cannot express it -- see this file's
// own target, tiny-4x4-8t.yaml). 64x4 weights over 4x4 tiles is 16 blocks
// on 8 tiles: the mm-spill-2x shape at test scale.
//
// WHY THIS FILE EXISTS, and it is not the obvious reason. The interesting
// number here is not that hoisting happens -- it is that the split is
// exactly 7 above the loop and 9 inside, which is blocks - (tiles - 1)
// programs per iteration. That is the proven optimum for ANY single loop
// body: a weight no op in the body programs is never written and so
// occupies a tile permanently, at most tiles-1 weights can be permanently
// resident, and everything else must be reprogrammed every iteration.
//
// cim-placement reaches that optimum, but it reaches it as a CONSEQUENCE
// of a tie-break rather than by aiming at it: Belady's victim scan starts
// at tile 0 and breaks immediately on a never-again next-use, so tile 0
// absorbs the entire spill and tiles 1..7 are each written exactly once --
// which is precisely the condition (programCountPerTile == 1) that puts a
// program in hoistCandidates. Change the victim selection to spread
// evictions across tiles -- a round-robin victim, a different scan order,
// dropping the early break -- and every tile's program count exceeds one,
// nothing is hoisted at all, and this workload silently regresses from
// 7 + 9*T programs to 16*T, which is exactly what a plain LRU cache would
// do. The flagship result would be gone with no other failing test in the
// suite, because the numerical suites cannot see it (a pass that hoists
// nothing computes identical values) and tiny-4x4.yaml's 2 tiles reach the
// same bound under any schedule.
//
// So this file is the guard on that. If the counts below change, the
// question to ask is not "is the test stale" but "did eviction stop
// concentrating, and is the flagship claim still true".

memref.global "private" constant @w : memref<64x4xi8> = dense<1>
memref.global "private" constant @acts : memref<4x4xi8> = dense<2>
func.func private @sink(memref<4x64xi32>)

// CHECK-LABEL: func.func @spill_loop
func.func @spill_loop() {
  %w = memref.get_global @w : memref<64x4xi8>
  %acts = memref.get_global @acts : memref<4x4xi8>
  %out = memref.alloc() : memref<4x64xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %actRow = memref.subview %acts[%i, 0] [1, 4] [1, 1]
      : memref<4x4xi8> to memref<1x4xi8, strided<[4, 1], offset: ?>>
    %actLocal = memref.alloc() : memref<1x4xi8>
    memref.copy %actRow, %actLocal
      : memref<1x4xi8, strided<[4, 1], offset: ?>> to memref<1x4xi8>
    %outRow = memref.subview %out[%i, 0] [1, 64] [1, 1]
      : memref<4x64xi32> to memref<1x64xi32, strided<[64, 1], offset: ?>>
    linalg.matmul_transpose_b ins(%actLocal, %w : memref<1x4xi8>, memref<64x4xi8>)
      outs(%outRow : memref<1x64xi32, strided<[64, 1], offset: ?>>)
    memref.dealloc %actLocal : memref<1x4xi8>
  }
  func.call @sink(%out) : (memref<4x64xi32>) -> ()
  return
}

// Seven hoisted above the loop -- tiles - 1, the most that can stay
// permanently resident -- and not an eighth.
// CHECK-COUNT-7: cim.program
// CHECK-NOT: cim.program
// CHECK: scf.for
// Nine left inside, one per iteration: blocks - (tiles - 1).
// CHECK-COUNT-9: cim.program
// CHECK-NOT: cim.program

// The cost report publishes both numbers and the distance between them,
// so the gap between what this pass emits and what an unrestricted
// N-inference Belady solve would achieve is a measured field in the
// artifact rather than a claim in a markdown file. Emitted is
// 7 + 9*4 = 43; the optimum over the flattened 4-iteration sequence is 40.
// COST: n-inference-optimum-programs: 40
// COST-NEXT: emitted-programs: 43
// COST-NEXT: placement-gap-percent: 7.50
