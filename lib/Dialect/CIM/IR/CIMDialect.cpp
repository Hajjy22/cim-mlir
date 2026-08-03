//===- CIMDialect.cpp - CIM dialect registration ---------------*- C++ -*-===//
//
// The generated type and attribute definitions live in this translation
// unit rather than CIMTypes.cpp, because initialize() below registers them
// with addTypes/addAttributes, which needs their storage classes to be
// complete. This is the same arrangement mlir/examples/standalone uses.
// CIMTypes.cpp then holds only the hand-written parse/print/verify members.
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect/CIMDialect.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Dialect/CIMTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cim;

#include "cim/Dialect/CIMEnums.cpp.inc"

#include "cim/Dialect/CIMOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cim/Dialect/CIMOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "cim/Dialect/CIMOpsTypes.cpp.inc"

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
