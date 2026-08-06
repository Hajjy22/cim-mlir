// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file | FileCheck %s

// cim-placement was unreachable dead code until this test existed: its use
// sequence was never populated, so the early return always fired and the
// engine was never called. What is checked here is the *structure* of the
// rewrite; test/mlir/pipeline_e2e_test.cpp checks that the rewritten IR
// still computes the same numbers, which matters more and cannot be
// expressed in FileCheck.

// Two matmuls sharing one weight global. tiny-4x4.yaml has 2 tiles of 4x4,
// and 8x4 weights are 2 blocks, so everything fits: the second matmul must
// reuse both residents rather than reprogramming them.

memref.global "private" constant @w : memref<8x4xi8> = dense<1>
memref.global "private" constant @a1 : memref<1x4xi8> = dense<2>
memref.global "private" constant @a2 : memref<1x4xi8> = dense<3>

func.func private @sink(memref<1x8xi32>)

// CHECK-LABEL: func.func @fits_in_tiles
func.func @fits_in_tiles() {
  %w = memref.get_global @w : memref<8x4xi8>

  %i1 = memref.get_global @a1 : memref<1x4xi8>
  %a1 = memref.alloc() : memref<1x4xi8>
  memref.copy %i1, %a1 : memref<1x4xi8> to memref<1x4xi8>
  %o1 = memref.alloc() : memref<1x8xi32>
  linalg.matmul_transpose_b ins(%a1, %w : memref<1x4xi8>, memref<8x4xi8>)
    outs(%o1 : memref<1x8xi32>)
  func.call @sink(%o1) : (memref<1x8xi32>) -> ()

  %i2 = memref.get_global @a2 : memref<1x4xi8>
  %a2 = memref.alloc() : memref<1x4xi8>
  memref.copy %i2, %a2 : memref<1x4xi8> to memref<1x4xi8>
  %o2 = memref.alloc() : memref<1x8xi32>
  linalg.matmul_transpose_b ins(%a2, %w : memref<1x4xi8>, memref<8x4xi8>)
    outs(%o2 : memref<1x8xi32>)
  func.call @sink(%o2) : (memref<1x8xi32>) -> ()

  return
}

// The first matmul programs both blocks. Tile ids come from the schedule
// and must be inside the target's declared count of 2 -- spec Sec. 5.4
// rule 5, which TileAllocOp::verify cannot enforce because it cannot see
// the target file.
// CHECK: cim.tile_alloc %{{.*}} {id = 0 : i64}
// CHECK: %[[R0:.*]] = cim.program {{.*}} -> !cim.resident<4x4xi8>
// CHECK: cim.mvm %[[R0]],
// CHECK: cim.tile_alloc %{{.*}} {id = 1 : i64}
// CHECK: %[[R1:.*]] = cim.program {{.*}} -> !cim.resident<4x4xi8>
// CHECK: cim.mvm %[[R1]],
// CHECK: call @sink

// After the first matmul, nothing is programmed again and no tile is
// re-allocated: the second matmul's mvms consume the residents already in
// the tiles. The CHECK-NOTs cover the whole region up to each following
// match, so a reintroduced cim.program anywhere in the second matmul fails.
// CHECK-NOT: cim.program
// CHECK-NOT: cim.tile_alloc
// CHECK: cim.mvm %[[R0]],
// CHECK-NOT: cim.program
// CHECK-NOT: cim.tile_alloc
// CHECK: cim.mvm %[[R1]],
// CHECK: call @sink

// -----

// A single matmul uses each of its blocks exactly once, so there is nothing
// to reuse and placement must not remove anything. This is the companion to
// the reuse case: a pass that eliminated programs too eagerly would pass the
// test above and fail here.

memref.global "private" constant @w2 : memref<16x4xi8> = dense<1>
memref.global "private" constant @act : memref<1x4xi8> = dense<2>

func.func private @sink2(memref<1x16xi32>)

// CHECK-LABEL: func.func @nothing_to_reuse
func.func @nothing_to_reuse() {
  %w = memref.get_global @w2 : memref<16x4xi8>
  %i = memref.get_global @act : memref<1x4xi8>
  %a = memref.alloc() : memref<1x4xi8>
  memref.copy %i, %a : memref<1x4xi8> to memref<1x4xi8>
  %o = memref.alloc() : memref<1x16xi32>
  linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<16x4xi8>)
    outs(%o : memref<1x16xi32>)
  func.call @sink2(%o) : (memref<1x16xi32>) -> ()
  return
}

// 16x4 over 4x4 tiles is 4 blocks on 2 tiles: four distinct weights, so all
// four programs survive and every id stays inside the declared count of 2.
//
// Which tile each block lands in is deliberately NOT pinned. After the first
// two steps every resident is dead -- no weight here is ever used twice --
// so the eviction choice is a tie and any of them is equally optimal. An
// exact-id CHECK would be asserting the engine's tie-break rather than any
// property of the schedule, and would break on a legitimate change to it.
// What must hold is rule 5: the id is a tile this device actually has.
// CHECK: cim.tile_alloc %{{.*}} {id = {{[01]}} : i64}
// CHECK: cim.program
// CHECK: cim.tile_alloc %{{.*}} {id = {{[01]}} : i64}
// CHECK: cim.program
// CHECK: cim.tile_alloc %{{.*}} {id = {{[01]}} : i64}
// CHECK: cim.program
// CHECK: cim.tile_alloc %{{.*}} {id = {{[01]}} : i64}
// CHECK: cim.program
// CHECK: call @sink2
