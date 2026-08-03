//===- cim-run.cpp - Execute cim IR and print its results ------*- C++ -*-===//
//
//   cim-opt model.mlir --cim-detect --cim-partition=target-yaml=T \
//     | cim-run --target-yaml=T -
//
// Runs the module's entry function through the cim interpreter against the
// real cimrt runtime, printing whatever the module asks for via its
// @cim_print_* calls. The in-process C++ test (test/mlir) is the day-to-day
// gate; this exists so the pipeline can be driven from a shell or from
// Python, which is what the numerical differential against numpy needs.
//
//===----------------------------------------------------------------------===//

#include "cim/Dialect/CIMDialect.h"
#include "cim/Interpreter/Interpreter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input .mlir file>"),
                  llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    targetYAML("target-yaml",
               llvm::cl::desc("Target description to open (spec Sec. 7)"),
               llvm::cl::init(""));

static llvm::cl::opt<std::string>
    entryName("entry", llvm::cl::desc("Entry function to execute"),
              llvm::cl::init("main"));

static llvm::cl::opt<bool>
    dumpProfile("profile",
                llvm::cl::desc("Print cimrt profile counters after execution"),
                llvm::cl::init(false));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "cim-mlir interpreter driver\n");

  if (targetYAML.empty()) {
    llvm::errs() << "cim-run: --target-yaml is required; the interpreter has "
                    "no device to execute against without it\n";
    return 1;
  }

  DialectRegistry registry;
  registry.insert<cim::CIMDialect, func::FuncDialect, memref::MemRefDialect,
                  arith::ArithDialect, linalg::LinalgDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "cim-run: could not open '" << inputFilename
                 << "': " << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagnostics(sourceMgr, &context);

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "cim-run: failed to parse the module\n";
    return 1;
  }

  cim::InterpreterOptions options;
  options.targetYAMLPath = targetYAML;
  options.out = &llvm::outs();
  options.dumpProfile = dumpProfile;

  if (failed(cim::run(*module, entryName, options))) {
    llvm::errs() << "cim-run: execution failed\n";
    return 1;
  }
  return 0;
}
