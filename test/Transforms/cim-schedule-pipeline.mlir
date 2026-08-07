// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-schedule | FileCheck %s

// cim-schedule must insert a barrier into IR cim-detect/cim-partition/
// cim-placement actually produce, not only into cim ops written directly
// in a test file (test/Transforms/cim-schedule.mlir covers the placement
// rules precisely with hand-written IR; this file is the "does it work on
// the real pipeline" check, the same division of labor as every other pass
// in this project).

// CHECK-LABEL: func.func @straight_line
func.func @straight_line() {
  %w = memref.get_global @w : memref<4x4xi8>
  %a = memref.alloc() : memref<1x4xi8>
  %out = memref.alloc() : memref<1x4xi32>
  linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<4x4xi8>)
                            outs(%out : memref<1x4xi32>)
  // CHECK: cim.program
  // CHECK: cim.mvm
  // CHECK: cim.barrier
  return
}
memref.global "private" constant @w : memref<4x4xi8> = dense<1>
