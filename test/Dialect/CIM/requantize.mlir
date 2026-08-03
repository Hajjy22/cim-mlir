// RUN: cim-opt %s | cim-opt | FileCheck %s

// CHECK-LABEL: func.func @requantize
func.func @requantize(%sum: memref<256xi32, #cim.space<near>>)
    -> memref<256xi8, #cim.space<near>> {
  // CHECK: cim.requantize
  %q = cim.requantize %sum {
         scale = 0.00392 : f32, zero_point = 0 : i32, effective_bits = 8 : i32
       } : memref<256xi32, #cim.space<near>> -> memref<256xi8, #cim.space<near>>
  return %q : memref<256xi8, #cim.space<near>>
}
