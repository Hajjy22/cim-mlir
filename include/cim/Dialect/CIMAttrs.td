//===- CIMAttrs.td - CIM dialect attributes ---------------*- tablegen -*-===//
//
// #cim.space<host|near|insitu>       - address space (spec Sec. 3.4)
// #cim.persistence<volatile|nonvolatile> - weight persistence class (spec Sec. 3.1/3.2)
//
//===----------------------------------------------------------------------===//

#ifndef CIM_ATTRS
#define CIM_ATTRS

include "cim/Dialect/CIMDialect.td"
include "mlir/IR/EnumAttr.td"

def CIM_Host   : I32EnumAttrCase<"host", 0>;
def CIM_Near   : I32EnumAttrCase<"near", 1>;
def CIM_Insitu : I32EnumAttrCase<"insitu", 2>;

def CIM_SpaceKind : I32EnumAttr<"SpaceKind", "cim address space", [
    CIM_Host, CIM_Near, CIM_Insitu]> {
  let cppNamespace = "::mlir::cim";
}

def CIM_Volatile    : I32EnumAttrCase<"volatile_", 0, "volatile">;
def CIM_Nonvolatile : I32EnumAttrCase<"nonvolatile", 1>;

def CIM_PersistenceKind : I32EnumAttr<"PersistenceKind", "cim weight persistence class", [
    CIM_Volatile, CIM_Nonvolatile]> {
  let cppNamespace = "::mlir::cim";
}

def CIM_SpaceAttr : CIM_Attr<"Space", "space"> {
  let summary = "Memory space a buffer lives in: host, near (tile-local scratchpad), or insitu (the weight array itself)";
  let parameters = (ins EnumParameter<CIM_SpaceKind>:$kind);
  let assemblyFormat = "`<` $kind `>`";
}

def CIM_PersistenceAttr : CIM_Attr<"Persistence", "persistence"> {
  let summary = "Whether resident weights survive across kernels / power-off";
  let parameters = (ins EnumParameter<CIM_PersistenceKind>:$kind);
  let assemblyFormat = "`<` $kind `>`";
}

#endif // CIM_ATTRS
