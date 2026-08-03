//===- cim-opt.cpp - cim-mlir optimizer driver -----------------*- C++ -*-===//
//
// The main dev tool (spec Sec. 12): registers the `cim` dialect and its
// 8-pass lowering pipeline (spec Sec. 6) alongside the standard upstream
// MLIR dialects/passes, so `cim-opt --cim-detect --cim-partition ...` can
// be run on .mlir input like any other MLIR pass pipeline.
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect/CIMDialect.h"
#include "cim/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::cim::CIMDialect>();

  mlir::registerAllPasses();
  mlir::cim::registerCIMPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "cim-mlir optimizer driver\n", registry));
}
