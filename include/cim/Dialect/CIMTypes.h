//===- CIMTypes.h - CIM dialect types -------------------------*- C++ -*-===//
#ifndef CIM_DIALECT_CIMTYPES_H
#define CIM_DIALECT_CIMTYPES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_ATTRDEF_CLASSES
#include "cim/Dialect/CIMOpsAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cim/Dialect/CIMOpsTypes.h.inc"

#endif // CIM_DIALECT_CIMTYPES_H
