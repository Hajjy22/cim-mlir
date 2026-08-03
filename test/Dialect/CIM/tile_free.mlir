// RUN: cim-opt %s | cim-opt | FileCheck %s

// CHECK-LABEL: func.func @free_tile
func.func @free_tile(%t: !cim.tile<256x256xi8>) {
  // CHECK: cim.tile_free
  cim.tile_free %t : !cim.tile<256x256xi8>
  return
}
