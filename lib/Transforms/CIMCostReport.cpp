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

#include <algorithm>
#include <cstdio>
#include <string>

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
          << counts.unknownProgramSites << " cim.program site(s), "
          << counts.unknownMvmSites << " cim.mvm site(s), "
          << counts.unknownRequantizeSites << " cim.requantize site(s), and "
          << counts.unknownReduceSites
          << " cim.reduce_partial site(s) sit under a loop whose trip count "
             "is not a compile-time constant; they are excluded from the "
             "totals below rather than guessed, so the report is a lower "
             "bound, not the full picture";

    ::cim::CostReport report;
    report.persistent = spec.tiles.persistent;
    // Copied from the target the same way computeCostReport
    // (lib/Placement/CostReport.cpp) does for the engine-side path. This
    // pass builds its CostReport by hand from IR op counts rather than
    // from a PlacementResult, so it does not go through that function and
    // has to mirror these assignments explicitly. Leaving them at their
    // struct defaults is not a harmless omission: toJson's
    // standby_leakage_pj_per_inference_floor is
    // numTiles * standbyLeakageUwPerTile * steadyStateElapsedNsPerInference,
    // so ANY of them left at zero silently reports a leak-free device, and
    // double_buffer_capable reports false for hardware whose own target
    // file declares the capability. cim-bench reads the same three values
    // straight from the spec and gets them right, so before this the
    // project shipped two emitters of one JSON schema that disagreed.
    report.numTiles = spec.tiles.count;
    report.standbyLeakageUwPerTile = spec.costs.standbyLeakageUwPerTile;
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

    report.requantizes = counts.requantizes;
    report.requantizeEnergyPj = static_cast<double>(counts.requantizes) *
                                 spec.costs.requantize.energyPj;
    report.requantizeLatencyNs = static_cast<double>(counts.requantizes) *
                                  spec.costs.requantize.latencyNs;

    report.reducePartialAdds = counts.reduceAdds;
    report.reducePartialEnergyPj = static_cast<double>(counts.reduceAdds) *
                                    spec.costs.reducePartial.energyPj;
    report.reducePartialLatencyNs = static_cast<double>(counts.reduceAdds) *
                                     spec.costs.reducePartial.latencyNs;

    // Charged against costs.reduce_max, NOT costs.reduce_partial above --
    // the shipped targets give the two entries deliberately different
    // values precisely so this distinction is visible in the numbers.
    report.reduceMaxes = counts.reduceMaxes;
    report.reduceMaxEnergyPj = static_cast<double>(counts.reduceMaxes) *
                                spec.costs.reduceMax.energyPj;
    report.reduceMaxLatencyNs = static_cast<double>(counts.reduceMaxes) *
                                 spec.costs.reduceMax.latencyNs;

    // Transfers: an entire declared cost class that this report used to
    // omit silently, while costs.transfer.energy_pj_per_byte sat in every
    // shipped target file as a required field. Only the host<->device
    // copies are charged -- see IRCostCounts::transferBytes for what is
    // deliberately still missing (implicit staging) and why no IR walk can
    // see it.
    report.transferBytes = counts.transferBytes;
    report.hostToHostCopies = counts.hostToHostCopies;
    report.transferEnergyPj = static_cast<double>(counts.transferBytes) *
                               spec.costs.transfer.energyPjPerByte;
    // bytes / (GB/s) == ns exactly, since 1 GB/s is 1 byte per ns. Mirrors
    // CostAccumulator::recordTransfer so the static and runtime sides read
    // bandwidth_gbps the same way; the units themselves are pinned by a
    // test, not just by these two comments agreeing.
    if (spec.costs.transfer.bandwidthGbps > 0.0)
      report.transferLatencyNs = static_cast<double>(counts.transferBytes) /
                                  spec.costs.transfer.bandwidthGbps;

    report.totalEnergyPj = report.programEnergyPj + report.mvmEnergyPj +
                            report.requantizeEnergyPj +
                            report.reducePartialEnergyPj +
                            report.reduceMaxEnergyPj +
                            report.transferEnergyPj;
    report.totalLatencyNs = report.programLatencyNs + report.mvmLatencyNs +
                             report.requantizeLatencyNs +
                             report.reducePartialLatencyNs +
                             report.reduceMaxLatencyNs +
                             report.transferLatencyNs;

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

    // Elapsed = reprogramming + compute, sequential, matching
    // computeCostReport's own steadyStateElapsedNsPerInference (and its
    // comment on why it stays a SUM even on a double-buffer-capable
    // target: that describes the hardware, not the schedule v0.1 emits).
    // The module this pass reports on bakes exactly one inference -- see
    // python/cim_frontend/onnx_import.py's own note, and the fact that
    // cim-run takes no runtime inputs -- so counts.mvms IS the per-
    // inference mvm count, no window division needed. This has to be set
    // for the leakage floor above to be non-zero on a volatile target.
    const double mvmLatencyPerInferenceNs =
        static_cast<double>(counts.mvms) * spec.costs.mvm.latencyNs;
    report.steadyStateElapsedNsPerInference =
        report.steadyStateLatencyNsPerInference + mvmLatencyPerInferenceNs;

    // A projection, computed only where the target says it could ever be
    // realized -- never substituted for the honest sum above. Same
    // discipline and same formula as computeCostReport's.
    report.doubleBufferCapable = spec.capabilities.doubleBufferProgram;
    if (report.doubleBufferCapable)
      report.steadyStateElapsedNsPerInferenceIfOverlapped =
          std::max(report.steadyStateLatencyNsPerInference,
                   mvmLatencyPerInferenceNs);

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
      out += ", unknown_requantize_sites=";
      out += std::to_string(counts.unknownRequantizeSites);
      out += ", unknown_reduce_sites=";
      out += std::to_string(counts.unknownReduceSites);
      out += ")";
    }
    out += "\n";

    // The N-inference optimum cim-placement recorded, if it computed one
    // (decision 4 in CIMPlacement.cpp's file header). Reported BESIDE what
    // the emitted IR actually costs, never instead of it: the whole point
    // is that these two numbers are allowed to differ and the difference
    // should be visible in the artifact rather than only in prose. An
    // absent attribute means "no loop body was solvable this way here",
    // which is a different statement from a zero gap, so the fields are
    // omitted entirely rather than printed as 0.
    if (auto opt = module->getAttrOfType<DictionaryAttr>(
            "cim.n_inference_optimum")) {
      if (auto programs = opt.getAs<IntegerAttr>("programs")) {
        const uint64_t optimal = static_cast<uint64_t>(programs.getInt());
        out += "// n-inference-optimum-programs: ";
        out += std::to_string(optimal);
        out += "\n// emitted-programs: ";
        out += std::to_string(counts.programs);
        // Only meaningful when the emitted count is itself complete --
        // an incomplete count is a lower bound, and a "gap" computed
        // against a lower bound would read as better than reality.
        if (counts.complete() && optimal > 0) {
          const double gap =
              100.0 * (static_cast<double>(counts.programs) -
                       static_cast<double>(optimal)) /
              static_cast<double>(optimal);
          char buf[64];
          std::snprintf(buf, sizeof(buf), "%.2f", gap);
          out += "\n// placement-gap-percent: ";
          out += buf;
        }
        out += "\n";
      }
    }

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
