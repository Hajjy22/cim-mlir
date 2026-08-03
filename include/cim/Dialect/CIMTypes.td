//===- CIMTypes.td - CIM dialect types ---------------------*- tablegen -*-===//
//
// !cim.tile<256x256xi8>        - a physical compute-and-storage tile (spec Sec. 3.2/5.1)
// !cim.resident<256x256xi8>    - a handle to weights currently resident in a tile
// !cim.device<"erbium-8t">     - a target-level device handle
//
// Tile/Resident use a shaped-type syntax ("RxCxElementType") like the
// builtin MemRefType/VectorType, which ODS's assemblyFormat mini-language
// cannot express directly — parsing/printing is implemented by hand in
// lib/Dialect/CIM/IR/CIMTypes.cpp (TODO, see that file).
//
//===----------------------------------------------------------------------===//

#ifndef CIM_TYPES
#define CIM_TYPES

include "cim/Dialect/CIMDialect.td"
include "mlir/IR/AttrTypeBase.td"

def CIM_TileType : CIM_Type<"Tile", "tile"> {
  let summary = "A fixed-capacity physical CIM/PIM compute tile";
  let description = [{
    Lowest common denominator hardware abstraction (spec Sec. 3.2): covers a
    UPMEM DPU, a digital SRAM CIM macro, an analog crossbar, or a near-memory
    RISC-V + iRAM cluster alike. Parameterized by geometry and weight element
    type, e.g. `!cim.tile<256x256xi8>`.
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def CIM_ResidentType : CIM_Type<"Resident", "resident"> {
  let summary = "A handle to weights currently resident (programmed) in a tile";
  let description = [{
    Returned by cim.program, consumed by cim.mvm. This is an SSA value, not a
    side effect on a mutable tile object (spec Sec. 5.4 verifier rule 3): a
    !cim.resident is invalidated the moment its tile is re-programmed, which
    is what lets cim-placement (Sec. 6) reason about residency purely from
    use-def chains.
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;
}

def CIM_DeviceType : CIM_Type<"Device", "device"> {
  let summary = "A handle to an open target device, e.g. !cim.device<\"erbium-8t\">";
  let parameters = (ins StringRefParameter<"target name">:$target);
  let hasCustomAssemblyFormat = 1;
}

#endif // CIM_TYPES
