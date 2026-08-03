//===- CIMDialect.td - CIM dialect definition -------------*- tablegen -*-===//
//
// Base dialect definition and op/type/attr base classes for the `cim`
// dialect. See docs/abstraction.md for the hardware model this dialect
// encodes (spec Section 3): CIM/PIM hardware is modeled as an array of
// fixed-capacity Tiles, addressed through three explicit memory spaces
// (host / near / insitu), with weight programming (cim.program) modeled
// as an expensive, non-foldable side-effecting op distinct from the cheap,
// repeatable compute op (cim.mvm).
//
//===----------------------------------------------------------------------===//

#ifndef CIM_DIALECT
#define CIM_DIALECT

include "mlir/IR/OpBase.td"

def CIM_Dialect : Dialect {
  let name = "cim";
  let summary = "Compute-in-Memory / Processing-in-Memory hardware dialect";
  let description = [{
    A hardware-agnostic dialect for compiling ML workloads onto CIM/PIM
    accelerators. Models hardware as an array of Tiles (fixed-capacity
    compute-and-storage units) addressed through explicit memory spaces.
    See docs/abstraction.md for the full model.
  }];
  let cppNamespace = "::mlir::cim";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Base classes every op/type/attr in this dialect derives from.
class CIM_Op<string mnemonic, list<Trait> traits = []> :
    Op<CIM_Dialect, mnemonic, traits>;

// The parameter is deliberately not called `mnemonic`: TableGen treats
// `let mnemonic = mnemonic` as recursive self-assignment and rejects it.
class CIM_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<CIM_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class CIM_Attr<string name, string attrMnemonic, list<Trait> traits = []> :
    AttrDef<CIM_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // CIM_DIALECT
