// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-digital-cim.yaml \
// RUN:   | FileCheck %s
//
// docs/roadmap.md's M4 "prove retargetability" entry: every other
// cim-partition test in this suite runs against a target declaring
// `persistent: true` (erbium-8t.yaml, tiny-4x4.yaml, tiny-4x4-4bit.yaml),
// so cim.program's own `persistent` attribute -- set directly from
// spec.tiles.persistent in lib/Transforms/CIMPartition.cpp -- had never
// been checked as `false` anywhere past the parser. This is that check,
// against tiny-digital-cim.yaml, which mirrors
// targets/generic-digital-cim.yaml's real volatile-SRAM characteristics
// at reduced scale.

// CHECK-LABEL: func.func @single_tile_matmul
func.func @single_tile_matmul(%act: memref<1x4xi8>, %out: memref<1x4xi32>) {
  %w = memref.get_global @weights_4x4 : memref<4x4xi8>
  // CHECK: cim.program {{.*}} persistent = false
  linalg.matmul_transpose_b ins(%act, %w : memref<1x4xi8>, memref<4x4xi8>)
                            outs(%out : memref<1x4xi32>)
  return
}
memref.global "private" constant @weights_4x4 : memref<4x4xi8> = dense<1>
