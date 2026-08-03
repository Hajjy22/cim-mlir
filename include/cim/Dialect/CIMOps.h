//===- CIMOps.h - CIM dialect operations -----------------------*- C++ -*-===//
#ifndef CIM_DIALECT_CIMOPS_H
#define CIM_DIALECT_CIMOPS_H

#include "cim/Dialect/CIMDialect.h"
#include "cim/Dialect/CIMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "cim/Dialect/CIMOps.h.inc"

#endif // CIM_DIALECT_CIMOPS_H
