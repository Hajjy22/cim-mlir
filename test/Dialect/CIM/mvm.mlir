// RUN: cim-opt %s | cim-opt | FileCheck %s

// CHECK-LABEL: func.func @mvm
func.func @mvm(%r: !cim.resident<256x256xi8>,
               %act: memref<256xi8, #cim.space<near>>)
    -> memref<256xi32, #cim.space<near>> {
  // CHECK: cim.mvm
  %out = cim.mvm %r, %act {accumulate = false}
         : (!cim.resident<256x256xi8>, memref<256xi8, #cim.space<near>>)
         -> memref<256xi32, #cim.space<near>>
  return %out : memref<256xi32, #cim.space<near>>
}
