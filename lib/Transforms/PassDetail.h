//===- PassDetail.h - CIM pass base-class definitions ----------*- C++ -*-===//
#ifndef CIM_TRANSFORMS_PASSDETAIL_H
#define CIM_TRANSFORMS_PASSDETAIL_H

// The generated pass base classes name ModuleOp unqualified.
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cim {

#define GEN_PASS_CLASSES
#include "cim/Transforms/Passes.h.inc"

} // namespace cim
} // namespace mlir

#endif // CIM_TRANSFORMS_PASSDETAIL_H
