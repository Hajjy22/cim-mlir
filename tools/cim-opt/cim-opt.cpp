//===- cim-opt.cpp - cim-mlir optimizer driver -----------------*- C++ -*-===//
//
// The main dev tool (spec Sec. 12): registers the `cim` dialect and its
// 8-pass lowering pipeline (spec Sec. 6) so that
// `cim-opt --cim-detect --cim-partition ...` can be run on .mlir input like
// any other MLIR pass pipeline.
//
// Registers a focused dialect set rather than calling registerAllDialects():
// an out-of-tree tool that pulls in every upstream dialect links all of
// MLIR (including the GPU, SPIR-V, and LLVM-target dialects) for no
// benefit. What is registered here is what cim IR is actually written
// against -- func/memref/arith for the IR the dialect consumes, and
// linalg/tensor/scf because cim-detect (Pass 1) pattern-matches
// linalg.matmul. Add a dialect here when a pass genuinely needs it.
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect/CIMDialect.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::cim::CIMDialect,
                  mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect,
                  mlir::scf::SCFDialect>();

  // Generic utility passes (canonicalize, cse, ...) are worth having when
  // debugging a pipeline by hand.
  mlir::registerTransformsPasses();
  mlir::cim::registerCIMPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "cim-mlir optimizer driver\n", registry));
}
