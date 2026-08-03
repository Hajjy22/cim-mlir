// RUN: cim-opt %s | cim-opt | FileCheck %s

// CHECK-LABEL: func.func @reduce_partial
func.func @reduce_partial(%p0: memref<256xi32, #cim.space<near>>,
                           %p1: memref<256xi32, #cim.space<near>>)
    -> memref<256xi32, #cim.space<near>> {
  // CHECK: cim.reduce_partial
  %sum = cim.reduce_partial %p0, %p1
         : (memref<256xi32, #cim.space<near>>, memref<256xi32, #cim.space<near>>)
         -> memref<256xi32, #cim.space<near>>
  return %sum : memref<256xi32, #cim.space<near>>
}
