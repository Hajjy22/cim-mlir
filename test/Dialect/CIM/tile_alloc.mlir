// RUN: cim-opt %s | cim-opt | FileCheck %s

// Note the functional type: !cim.device carries a target-name parameter, so
// the operand type cannot be inferred and must appear in the syntax. See the
// description on CIM_TileAllocOp in include/cim/Dialect/CIMOps.td.

// CHECK-LABEL: func.func @alloc_tile
func.func @alloc_tile(%dev: !cim.device<"erbium-8t">) -> !cim.tile<256x256xi8> {
  // CHECK: cim.tile_alloc {{.*}} -> !cim.tile<256x256xi8>
  %t = cim.tile_alloc %dev {id = 0 : i64}
       : (!cim.device<"erbium-8t">) -> !cim.tile<256x256xi8>
  return %t : !cim.tile<256x256xi8>
}
