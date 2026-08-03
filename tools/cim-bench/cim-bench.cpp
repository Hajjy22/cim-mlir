//===- cim-bench.cpp - Benchmark harness CLI (spec Sec. 10) -----*- C++ -*-===//
//
// `cim-bench run --target erbium-8t --out results.json`
//
// Harness requirements (spec Sec. 10): one command runs everything; every
// result carries the target file hash, git commit, and date; every plot
// script lives in the repo; every workload gets a correctness check against
// a PyTorch reference alongside its performance number.
//
// This is a v0.1 skeleton: it accepts the real CLI shape and writes a
// valid-but-empty results file so the contract is testable now, before the
// cim-opt pipeline (detect/partition/placement/...) is functional enough to
// actually run the spec Sec. 10 workload table (mm-fit, mm-spill-2x,
// mm-spill-8x, mlp-3layer, bert-ffn).
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::OptionCategory CimBenchCategory("cim-bench options");

static cl::opt<std::string>
    RunCommand(cl::Positional, cl::desc("<command>"), cl::init("run"),
               cl::cat(CimBenchCategory));

static cl::opt<std::string>
    TargetName("target", cl::desc("Target name, e.g. erbium-8t (spec Sec. 7)"),
               cl::init(""), cl::cat(CimBenchCategory));

static cl::opt<std::string> OutPath("out", cl::desc("Path to write results.json"),
                                     cl::init("results.json"),
                                     cl::cat(CimBenchCategory));

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(CimBenchCategory);
  cl::ParseCommandLineOptions(argc, argv, "cim-bench: cim-mlir benchmark harness\n");

  if (RunCommand != "run") {
    errs() << "cim-bench: unknown command '" << RunCommand << "' (only 'run' is implemented)\n";
    return 1;
  }
  if (TargetName.empty()) {
    errs() << "cim-bench: --target is required\n";
    return 1;
  }

  errs() << "cim-bench: not yet implemented (M0/M1 skeleton) -- the "
         << "spec Sec. 10 workload table (mm-fit, mm-spill-2x, mm-spill-8x, "
         << "mlp-3layer, bert-ffn) requires a working cim-opt pipeline.\n";

  std::error_code ec;
  ToolOutputFile out(OutPath, ec, sys::fs::OF_Text);
  if (ec) {
    errs() << "cim-bench: failed to open '" << OutPath << "': " << ec.message() << "\n";
    return 1;
  }

  // TODO(spec Sec.10): populate target file hash, git commit, and date per
  // the harness requirements, and one entry per executed workload.
  out.os() << "{\n"
            << "  \"target\": \"" << TargetName << "\",\n"
            << "  \"workloads\": []\n"
            << "}\n";
  out.keep();
  return 0;
}
