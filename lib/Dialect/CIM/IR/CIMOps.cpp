//===- CIMOps.cpp - CIM dialect op implementations -------------*- C++ -*-===//

#include "cim/Dialect/CIMOps.h"

using namespace mlir;
using namespace mlir::cim;

//===----------------------------------------------------------------------===//
// ProgramOp
//===----------------------------------------------------------------------===//

LogicalResult ProgramOp::verify() {
  // TODO(spec Sec.5.4 rule 3): the single rule that catches real bugs.
  // A !cim.resident produced by an earlier cim.program on the SAME tile
  // must have no remaining uses once this op re-programs that tile —
  // requires walking the tile's use-def chain (or a dedicated liveness
  // analysis shared with cim-placement) rather than a purely local check.
  return success();
}

//===----------------------------------------------------------------------===//
// MvmOp
//===----------------------------------------------------------------------===//

LogicalResult MvmOp::verify() {
  // TODO(spec Sec.5.4 rule 1): operand shapes must match the !cim.resident
  // geometry exactly.
  // TODO(spec Sec.5.4 rule 2): $activations must be in #cim.space<near>;
  // no implicit host access. Requires inspecting the memref's memory-space
  // attribute once cim.copy / space-typed memrefs are wired through.
  return success();
}

//===----------------------------------------------------------------------===//
// ReducePartialOp
//===----------------------------------------------------------------------===//

LogicalResult ReducePartialOp::verify() {
  // TODO(spec Sec.5.4 rule 4): all $partials operands must have identical
  // shape and be i32.
  return success();
}

#define GET_OP_CLASSES
#include "cim/Dialect/CIMOps.cpp.inc"
