// RUN: cim-opt %s | cim-opt | FileCheck %s

// CHECK-LABEL: func.func @program_tile
func.func @program_tile(%tile: !cim.tile<256x256xi8>,
                         %weights: memref<256x256xi8, #cim.space<host>>)
    -> !cim.resident<256x256xi8> {
  // CHECK: cim.program
  %r = cim.program %tile, %weights {
         cost_ns = 12000 : i64, cost_pj = 480000 : i64, persistent = true
       } : (!cim.tile<256x256xi8>, memref<256x256xi8, #cim.space<host>>)
         -> !cim.resident<256x256xi8>
  return %r : !cim.resident<256x256xi8>
}
