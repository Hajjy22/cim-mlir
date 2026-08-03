//===- CIMEnums.h - CIM dialect enums --------------------------*- C++ -*-===//
//
// Wraps the TableGen-generated enum declarations in a real include guard.
// The generated .inc has none, so both CIMDialect.h and CIMTypes.h include
// this rather than the .inc directly -- the attribute classes generated
// from CIMAttrs.td reference SpaceKind/PersistenceKind, so the enums must
// be visible first, and without a guard that means a redefinition error.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_DIALECT_CIMENUMS_H
#define CIM_DIALECT_CIMENUMS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

#include "cim/Dialect/CIMEnums.h.inc"

#endif // CIM_DIALECT_CIMENUMS_H
