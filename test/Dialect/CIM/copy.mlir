// RUN: cim-opt %s | cim-opt | FileCheck %s

// CHECK-LABEL: func.func @copy_host_to_near
func.func @copy_host_to_near(%src: memref<256xi8, #cim.space<host>>)
    -> memref<256xi8, #cim.space<near>> {
  // CHECK: cim.copy
  %d = cim.copy %src : memref<256xi8, #cim.space<host>>
                    to memref<256xi8, #cim.space<near>>
  return %d : memref<256xi8, #cim.space<near>>
}
