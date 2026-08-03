//===- CIMOps.cpp - CIM dialect op implementations -------------*- C++ -*-===//
//
// Verifier rules from spec Sec. 5.4. Rules 1 and 4 are enforced here; rules
// 2, 3, and 5 need information a single op cannot see and are marked as
// such below rather than being silently skipped.
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect/CIMOps.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::cim;

namespace {

/// Element type of a memref operand, or null if it is not a memref.
Type memRefElementType(Value value) {
  if (auto memref = llvm::dyn_cast<MemRefType>(value.getType()))
    return memref.getElementType();
  return nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// TileAllocOp
//===----------------------------------------------------------------------===//

LogicalResult TileAllocOp::verify() {
  // ODS gives an I64Attr an *unsigned* accessor, so `getId() < 0` is dead
  // code that always evaluates false -- a negative id written in the IR
  // (`id = -1 : i64`) round-trips as a huge unsigned and would sail past it.
  // Reinterpreting as signed is what actually rejects it.
  if (static_cast<int64_t>(getId()) < 0)
    return emitOpError("tile id must be non-negative");

  // Rule 5 (spec Sec. 5.4) also requires tile IDs to be unique and within
  // the target's declared tile count. Neither is checkable from a single
  // op: uniqueness is a module-wide property and the tile count comes from
  // the target file, which is not threaded into the verifier. cim-placement
  // enforces both once it owns tile assignment.
  return success();
}

//===----------------------------------------------------------------------===//
// ProgramOp
//===----------------------------------------------------------------------===//

LogicalResult ProgramOp::verify() {
  auto tileType = getTile().getType();
  auto residentType = getResult().getType();

  // The resident handle must describe exactly the tile it was programmed
  // into -- otherwise a later cim.mvm would compute against a geometry the
  // hardware does not have.
  if (tileType.getShape() != residentType.getShape())
    return emitOpError("resident shape ")
           << residentType.getShape() << " does not match tile shape "
           << tileType.getShape();
  if (tileType.getElementType() != residentType.getElementType())
    return emitOpError("resident element type does not match tile element type");

  // The weights being programmed must fill the tile.
  auto weightsType = llvm::dyn_cast<MemRefType>(getWeights().getType());
  if (!weightsType)
    return emitOpError("weights operand must be a memref");
  if (weightsType.getShape() != tileType.getShape())
    return emitOpError("weights shape ")
           << weightsType.getShape() << " does not match tile shape "
           << tileType.getShape();
  if (weightsType.getElementType() != tileType.getElementType())
    return emitOpError("weights element type does not match tile element type");

  // Rule 3 (spec Sec. 5.4) -- a !cim.resident may not be used after its
  // tile is re-programmed -- is deliberately not checked here. Because
  // cim.program returns a *new* resident handle rather than mutating the
  // tile, the stale-use case is a use-def property of the surrounding
  // region, not of this op. cim-placement is what establishes and relies on
  // that invariant.
  return success();
}

//===----------------------------------------------------------------------===//
// MvmOp
//===----------------------------------------------------------------------===//

LogicalResult MvmOp::verify() {
  auto residentType = getWeights().getType();
  auto actType = llvm::dyn_cast<MemRefType>(getActivations().getType());
  auto outType = llvm::dyn_cast<MemRefType>(getResult().getType());

  if (!actType)
    return emitOpError("activations operand must be a memref");
  if (!outType)
    return emitOpError("result must be a memref");

  // Rule 1: operand shapes must match the resident geometry exactly.
  // Weights are [rows x cols]; the activation vector is [cols] and the
  // output accumulator is [rows].
  auto shape = residentType.getShape();
  if (shape.size() != 2)
    return emitOpError("resident weights must be 2-D");
  const int64_t rows = shape[0];
  const int64_t cols = shape[1];

  if (actType.getRank() != 1 || actType.getShape()[0] != cols)
    return emitOpError("activations must be a 1-D memref of ")
           << cols << " elements to match the resident weights";
  if (outType.getRank() != 1 || outType.getShape()[0] != rows)
    return emitOpError("result must be a 1-D memref of ")
           << rows << " elements to match the resident weights";

  // Partial sums must not silently saturate (spec Sec. 5.3): the
  // accumulator has to be wider than the weight element type.
  auto weightElem = llvm::dyn_cast<IntegerType>(residentType.getElementType());
  auto outElem = llvm::dyn_cast<IntegerType>(outType.getElementType());
  if (weightElem && outElem && outElem.getWidth() <= weightElem.getWidth())
    return emitOpError("accumulator type ")
           << outElem << " is not wider than the weight type " << weightElem
           << "; partial sums would saturate";

  // Rule 2 -- activations must live in #cim.space<near>, no implicit host
  // access -- is not enforced yet: cim-insert-transfers is what attaches
  // space attributes to these memrefs, and until it does, requiring one
  // here would reject every hand-written test.
  return success();
}

//===----------------------------------------------------------------------===//
// ReducePartialOp
//===----------------------------------------------------------------------===//

LogicalResult ReducePartialOp::verify() {
  if (getPartials().empty())
    return emitOpError("expects at least one partial sum to reduce");

  // Rule 4: all operands must have identical shape, and be an integer
  // accumulator type -- reducing i8 partial sums would defeat the point of
  // accumulating in i32 in the first place.
  auto firstType = llvm::dyn_cast<MemRefType>(getPartials().front().getType());
  if (!firstType)
    return emitOpError("partial sums must be memrefs");

  for (Value partial : getPartials()) {
    auto type = llvm::dyn_cast<MemRefType>(partial.getType());
    if (!type)
      return emitOpError("partial sums must be memrefs");
    if (type.getShape() != firstType.getShape())
      return emitOpError("all partial sums must have identical shape");
    if (type.getElementType() != firstType.getElementType())
      return emitOpError("all partial sums must have identical element type");
  }

  if (!llvm::isa<IntegerType>(firstType.getElementType()))
    return emitOpError("partial sums must have an integer accumulator type");

  auto resultType = llvm::dyn_cast<MemRefType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a memref");
  if (resultType.getShape() != firstType.getShape())
    return emitOpError("result shape must match the partial sums");
  if (resultType.getElementType() != firstType.getElementType())
    return emitOpError("result element type must match the partial sums");

  return success();
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

LogicalResult CopyOp::verify() {
  auto srcType = llvm::dyn_cast<MemRefType>(getSource().getType());
  auto dstType = llvm::dyn_cast<MemRefType>(getDest().getType());
  if (!srcType || !dstType)
    return emitOpError("source and destination must be memrefs");

  // A copy moves bytes between address spaces; it must not silently reshape
  // or reinterpret them.
  if (srcType.getShape() != dstType.getShape())
    return emitOpError("copy must not change shape");
  if (srcType.getElementType() != dstType.getElementType())
    return emitOpError("copy must not change element type");
  if (srcType.getMemorySpace() == dstType.getMemorySpace())
    return emitOpError("source and destination are in the same memory space; "
                       "the copy has no effect");

  return success();
}

//===----------------------------------------------------------------------===//
// RequantizeOp
//===----------------------------------------------------------------------===//

LogicalResult RequantizeOp::verify() {
  auto inType = llvm::dyn_cast<MemRefType>(getInput().getType());
  auto outType = llvm::dyn_cast<MemRefType>(getResult().getType());
  if (!inType || !outType)
    return emitOpError("input and result must be memrefs");
  if (inType.getShape() != outType.getShape())
    return emitOpError("requantize must not change shape");

  // Signed reinterpretation for the same reason as TileAllocOp's id: the
  // ODS accessor for an I32Attr is unsigned, so a negative effective_bits
  // in the IR would otherwise read as a very large positive one.
  const int64_t effectiveBits = static_cast<int32_t>(getEffectiveBits());
  if (effectiveBits <= 0)
    return emitOpError("effective_bits must be positive");

  // effective_bits models what the readout path can actually resolve (an
  // ADC on an analog target). Claiming more bits out than the result type
  // can hold is a target-description error worth catching early.
  if (auto outElem = llvm::dyn_cast<IntegerType>(outType.getElementType())) {
    if (effectiveBits > static_cast<int64_t>(outElem.getWidth()))
      return emitOpError("effective_bits (")
             << getEffectiveBits() << ") exceeds the width of the result type "
             << outElem;
  }

  (void)memRefElementType(getInput());
  return success();
}

#define GET_OP_CLASSES
#include "cim/Dialect/CIMOps.cpp.inc"
