//===- CIMDialect.cpp - CIM dialect registration ---------------*- C++ -*-===//

#include "cim/Dialect/CIMDialect.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Dialect/CIMTypes.h"

using namespace mlir;
using namespace mlir::cim;

#include "cim/Dialect/CIMOpsDialect.cpp.inc"

void CIMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cim/Dialect/CIMOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "cim/Dialect/CIMOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "cim/Dialect/CIMOpsAttributes.cpp.inc"
      >();
}
