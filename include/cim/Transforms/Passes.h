//===- Passes.h - CIM lowering pipeline pass declarations ------*- C++ -*-===//
#ifndef CIM_TRANSFORMS_PASSES_H
#define CIM_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace cim {

std::unique_ptr<Pass> createCIMDetectPass();
std::unique_ptr<Pass> createCIMPartitionPass();
std::unique_ptr<Pass> createCIMPlacementPass();
std::unique_ptr<Pass> createCIMSchedulePass();
std::unique_ptr<Pass> createCIMInsertTransfersPass();
std::unique_ptr<Pass> createCIMLegalizePrecisionPass();
std::unique_ptr<Pass> createCIMLowerToTargetPass();
std::unique_ptr<Pass> createCIMCostReportPass();

#define GEN_PASS_REGISTRATION
#include "cim/Transforms/Passes.h.inc"

} // namespace cim
} // namespace mlir

#endif // CIM_TRANSFORMS_PASSES_H
