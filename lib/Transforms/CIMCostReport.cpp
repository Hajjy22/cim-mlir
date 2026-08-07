//===- CIMCostReport.cpp - Pass 8: emit JSON cost report -------*- C++ -*-===//
//
// Not an optimization pass -- this is the pass that produces the project's
// publishable numbers (spec Sec. 17 worked example).
//
// Reuses cim::CostReport / cim::toJson / cim::formatCostReport
// (lib/Placement/CostReport.cpp) for the actual arithmetic and the JSON/text
// format, rather than a second cost path that could drift from the one
// cim-bench and cimrt's profiler already use. The op-counting itself lives
// in CostReportUtils.cpp (cim::countWeightedOps), split out so tests can
// call it directly instead of capturing this pass's printed output.
//
// THE DESIGN POINT: a hoisted cim.program (cim-placement, spec Sec. 6 Pass
// 3) executes once regardless of where it sits textually; an op still
// inside an scf.for executes trip-count times. A plain walk-and-count would
// silently under- or over-report exactly the IR cim-placement now produces,
// which is why countWeightedOps weights every op by the product of its
// enclosing loops' compile-time-constant trip counts rather than just
// counting it.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Placement/CostReport.h"
#include "cim/Target/TargetSpec.h"
#include "cim/Transforms/CostReportUtils.h"
#include "cim/Transforms/Passes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::cim;

namespace {

struct CIMCostReportPass : public CIMCostReportBase<CIMCostReportPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (targetYAML.empty()) {
      module.emitError(
          "cim-cost-report requires -target-yaml=<path>; the cost table "
          "(energy_pj/latency_ns per program and mvm) comes from the "
          "target description (spec Sec. 7)");
      return signalPassFailure();
    }
    ::cim::TargetSpec spec;
    std::string parseError;
    if (!::cim::parseTargetSpecFromFile(targetYAML, spec, &parseError)) {
      module.emitError("cim-cost-report: ") << parseError;
      return signalPassFailure();
    }

    const IRCostCounts counts = countWeightedOps(module);

    if (!counts.complete())
      module.emitWarning("cim-cost-report: ")
          << counts.unknownProgramSites << " cim.program site(s) and "
          << counts.unknownMvmSites
          << " cim.mvm site(s) sit under a loop whose trip count is not a "
             "compile-time constant; they are excluded from the totals "
             "below rather than guessed, so the report is a lower bound, "
             "not the full picture";

    ::cim::CostReport report;
    report.persistent = spec.tiles.persistent;
    report.programs = counts.programs;
    report.mvms = counts.mvms;
    // Every cim.mvm reads one resident tile; a step that needed a fresh
    // cim.program is not a reuse, everything else is. This is derived, not
    // independently tracked, because the final IR after cim-placement no
    // longer carries a separate "was this step a reuse" marker -- reuse
    // steps are exactly the mvms with no corresponding program.
    report.reuses =
        counts.mvms >= counts.programs ? counts.mvms - counts.programs : 0;

    report.programEnergyPj =
        static_cast<double>(counts.programs) * spec.costs.program.energyPj;
    report.programLatencyNs =
        static_cast<double>(counts.programs) * spec.costs.program.latencyNs;
    report.mvmEnergyPj =
        static_cast<double>(counts.mvms) * spec.costs.mvm.energyPj;
    report.mvmLatencyNs =
        static_cast<double>(counts.mvms) * spec.costs.mvm.latencyNs;
    report.totalEnergyPj = report.programEnergyPj + report.mvmEnergyPj;
    report.totalLatencyNs = report.programLatencyNs + report.mvmLatencyNs;

    report.installPrograms = counts.installPrograms;
    report.installEnergyPj = static_cast<double>(counts.installPrograms) *
                              spec.costs.program.energyPj;
    report.installLatencyNs = static_cast<double>(counts.installPrograms) *
                               spec.costs.program.latencyNs;

    report.steadyStateProgramsPerInference = counts.steadyStatePrograms;
    report.steadyStateEnergyPjPerInference =
        static_cast<double>(counts.steadyStatePrograms) *
        spec.costs.program.energyPj;
    report.steadyStateLatencyNsPerInference =
        static_cast<double>(counts.steadyStatePrograms) *
        spec.costs.program.latencyNs;

    const std::string label =
        module.getName() ? module.getName()->str() : std::string("module");
    std::string json = ::cim::toJson(report, label);

    std::string out;
    out += "// trip-count-complete: ";
    out += counts.complete() ? "true" : "false";
    if (!counts.complete()) {
      out += " (unknown_program_sites=";
      out += std::to_string(counts.unknownProgramSites);
      out += ", unknown_mvm_sites=";
      out += std::to_string(counts.unknownMvmSites);
      out += ")";
    }
    out += "\n";
    out += json;

    if (outputPath.empty()) {
      llvm::outs() << out;
      return;
    }
    std::error_code ec;
    llvm::ToolOutputFile file(outputPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError("cim-cost-report: could not open '")
          << outputPath << "': " << ec.message();
      return signalPassFailure();
    }
    file.os() << out;
    file.keep();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMCostReportPass() {
  return std::make_unique<CIMCostReportPass>();
}
