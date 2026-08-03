//===- CIMTypes.cpp - CIM dialect type implementations ---------*- C++ -*-===//
//
// !cim.tile / !cim.resident use a MemRefType-style shaped syntax
// ("256x256xi8") that ODS's assemblyFormat mini-language can't express, so
// parsing/printing is hand-written here, following the same pattern
// MLIR's own builtin MemRefType/VectorType use.
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect/CIMTypes.h"

// The generated type/attribute definitions below reference CIMDialect and
// construct mlir::Builder, so both must be complete here.
#include "cim/Dialect/CIMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cim;

// The generated definitions themselves live in CIMDialect.cpp -- see the
// note at the top of that file. This TU holds only the members ODS asks us
// to write by hand.

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

Type TileType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};
  SmallVector<int64_t> shape;
  if (parser.parseDimensionList(shape, /*allowDynamic=*/false))
    return {};
  Type elementType;
  if (parser.parseType(elementType) || parser.parseGreater())
    return {};
  return TileType::get(parser.getContext(), shape, elementType);
}

void TileType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t dim : getShape())
    printer << dim << "x";
  printer << getElementType() << ">";
}

LogicalResult TileType::verify(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<int64_t> shape, Type elementType) {
  // TODO(spec Sec.5.4 rule 5): tile ID uniqueness / target tile-count bound
  // is an op-level (cim.tile_alloc) concern, not a type-level one; this
  // verifier only checks the type is well-formed as a 2D tile.
  if (shape.size() != 2)
    return emitError() << "cim.tile expects a 2D shape (rows x cols)";
  for (int64_t dim : shape)
    if (dim <= 0)
      return emitError() << "cim.tile dimensions must be positive";
  return success();
}

//===----------------------------------------------------------------------===//
// ResidentType
//===----------------------------------------------------------------------===//

Type ResidentType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};
  SmallVector<int64_t> shape;
  if (parser.parseDimensionList(shape, /*allowDynamic=*/false))
    return {};
  Type elementType;
  if (parser.parseType(elementType) || parser.parseGreater())
    return {};
  return ResidentType::get(parser.getContext(), shape, elementType);
}

void ResidentType::print(AsmPrinter &printer) const {
  printer << "<";
  for (int64_t dim : getShape())
    printer << dim << "x";
  printer << getElementType() << ">";
}

//===----------------------------------------------------------------------===//
// DeviceType
//===----------------------------------------------------------------------===//

Type DeviceType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};
  std::string target;
  if (parser.parseString(&target) || parser.parseGreater())
    return {};
  return DeviceType::get(parser.getContext(), target);
}

void DeviceType::print(AsmPrinter &printer) const {
  printer << "<\"" << getTarget() << "\">";
}
