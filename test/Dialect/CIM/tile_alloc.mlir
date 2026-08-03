// RUN: cim-opt %s | cim-opt | FileCheck %s

// CHECK-LABEL: func.func @alloc_tile
func.func @alloc_tile(%dev: !cim.device<"erbium-8t">) -> !cim.tile<256x256xi8> {
  // CHECK: cim.tile_alloc
  %t = cim.tile_alloc %dev {id = 0 : i64} : !cim.tile<256x256xi8>
  return %t : !cim.tile<256x256xi8>
}
