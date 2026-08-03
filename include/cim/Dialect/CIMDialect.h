//===- CIMDialect.h - CIM dialect declaration ------------------*- C++ -*-===//
#ifndef CIM_DIALECT_CIMDIALECT_H
#define CIM_DIALECT_CIMDIALECT_H

#include "mlir/IR/Dialect.h"

// Pulls in the CIM_SpaceKind / CIM_PersistenceKind enum classes generated
// from CIMAttrs.td.
#include "cim/Dialect/CIMEnums.h.inc"

#include "cim/Dialect/CIMOpsDialect.h.inc"

#endif // CIM_DIALECT_CIMDIALECT_H
