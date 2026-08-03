//===- PassDetail.h - CIM pass base-class definitions ----------*- C++ -*-===//
#ifndef CIM_TRANSFORMS_PASSDETAIL_H
#define CIM_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cim {

#define GEN_PASS_CLASSES
#include "cim/Transforms/Passes.h.inc"

} // namespace cim
} // namespace mlir

#endif // CIM_TRANSFORMS_PASSDETAIL_H
