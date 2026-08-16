//===- CostReport.h - Analytical cost of a placement -----------*- C++ -*-===//
//
// Turns a PlacementResult plus a TargetSpec into the numbers that make the
// argument for this project (spec Sec. 17): install-time weight programming
// amortized against per-inference compute.
//
// Deliberately analytical, not cycle-accurate (spec Sec. 9.2). Pure C++ so
// it can be tested without an LLVM toolchain.
//
// UNITS: everything here is picojoules and nanoseconds, matching the target
// file's `energy_pj` / `latency_ns` field names. See docs/target-format.md
// for a note on a units discrepancy in the spec's own worked example.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_PLACEMENT_COSTREPORT_H
#define CIM_PLACEMENT_COSTREPORT_H

#include "cim/Placement/Placement.h"
#include "cim/Target/TargetSpec.h"

#include <cstdint>
#include <string>

namespace cim {

struct CostReport {
  uint64_t programs = 0;
  uint64_t mvms = 0;
  uint64_t reuses = 0;

  double programEnergyPj = 0.0;
  double programLatencyNs = 0.0;
  double mvmEnergyPj = 0.0;
  double mvmLatencyNs = 0.0;

  /// cim.requantize is charged here, by cim-cost-report
  /// (lib/Transforms/CIMCostReport.cpp), which walks real compiled IR.
  /// This engine's own computeCostReport() below leaves these at their
  /// zero defaults: it models weight-programming amortization from a
  /// PlacementResult alone, which has no notion of a requantize step, so
  /// there is nothing for it to count here.
  uint64_t requantizes = 0;
  double requantizeEnergyPj = 0.0;
  double requantizeLatencyNs = 0.0;

  /// Same story as requantizes immediately above, for cim.reduce_partial's
  /// chained cimrt_reduce_add calls: charged only by cim-cost-report, left
  /// at zero here for the same reason (a PlacementResult alone has no
  /// notion of a reduce step either).
  uint64_t reducePartialAdds = 0;
  double reducePartialEnergyPj = 0.0;
  double reducePartialLatencyNs = 0.0;

  /// Bytes moved by cim.copy sites that cross the host/device boundary,
  /// and the energy `costs.transfer.energy_pj_per_byte` charges for them.
  /// Same "only cim-cost-report can count this" story as requantizes and
  /// reducePartialAdds above: a PlacementResult has no notion of a copy
  /// step, so the engine-side path leaves these at zero.
  ///
  /// IMPORTANT, and disclosed in the emitted report itself: this is NOT
  /// every byte the runtime charges. lowerProgram/lowerMvm stage weights,
  /// activations and results through cimrt_write/cimrt_read without any
  /// cim.copy op to represent them, and on a small placed module that
  /// implicit traffic is the majority of cimrt_profile's `bytes`. See
  /// IRCostCounts::transferBytes for the full reasoning and why the
  /// static-vs-runtime differential deliberately does not assert these
  /// two numbers are equal.
  uint64_t transferBytes = 0;
  double transferEnergyPj = 0.0;
  /// Time those bytes take at the target's declared bandwidth. `gbps` is
  /// read as GIGABYTES per second, so this is simply bytes / gbps -- see
  /// CostAccumulator::recordTransfer (runtime/src/simulator/cost_model.h)
  /// for why that reading, and not gigabits, is the right one, and
  /// transfer_latency_pins_the_gigabytes_per_second_convention in
  /// test/unit/cost_report_test.cpp for the test that holds it there.
  double transferLatencyNs = 0.0;
  /// Host-to-host cim.copy sites: real movement that neither this report
  /// nor cimrt charges, surfaced so zero-because-none is distinguishable
  /// from zero-because-uncharged.
  uint64_t hostToHostCopies = 0;

  double totalEnergyPj = 0.0;
  double totalLatencyNs = 0.0;

  /// Programs emitted during the first pass over the model. On a
  /// non-volatile, weight-stationary target these happen once at install
  /// time and never again.
  uint64_t installPrograms = 0;
  double installEnergyPj = 0.0;
  double installLatencyNs = 0.0;

  /// Programs emitted during the final modeled pass — i.e. what a steady
  /// state inference actually costs once the tiles have warmed up. Zero
  /// here is the headline result: the model fits, so inference reprograms
  /// nothing.
  uint64_t steadyStateProgramsPerInference = 0;
  double steadyStateEnergyPjPerInference = 0.0;
  double steadyStateLatencyNsPerInference = 0.0;

  /// Total analytical wall-clock time (spec Sec. 9.2) one steady-state
  /// inference takes -- steadyStateLatencyNsPerInference's reprogramming
  /// time plus that same window's mvm time, which nothing above needed to
  /// combine before. Exists for amortizedInstallEnergyPjPerInference: a
  /// volatile array draws standby leakage for however long the array stays
  /// powered during an inference, whether it is reprogramming or computing,
  /// not just the reprogramming slice steadyStateLatencyNsPerInference
  /// tracks on its own.
  ///
  /// This is a SUM of the window's program and mvm latencies -- strictly
  /// sequential, reprogram-then-compute, with no overlap between the two --
  /// which is the honest number for what this compiler actually emits
  /// today: `cim-schedule`'s own header states plainly that "nothing is
  /// reordered or overlapped" (v0.1 scope). It stays the sum on EVERY
  /// target, including one that declares `capabilities.double_buffer_
  /// program`, because that capability describes what the HARDWARE could
  /// do, not what the CURRENT compiled schedule does with it.
  double steadyStateElapsedNsPerInference = 0.0;

  /// Copied from the target file (`capabilities.double_buffer_program`):
  /// can this hardware reprogram tile i+1 while computing on tile i? A
  /// hardware capability, not something v0.1 exploits -- see
  /// steadyStateElapsedNsPerInference's own comment.
  bool doubleBufferCapable = false;

  /// A PROJECTION, not a measurement of anything this compiler emits: what
  /// steadyStateElapsedNsPerInference WOULD be if a scheduler exploited
  /// doubleBufferCapable to pipeline reprogramming against compute across
  /// tiles -- max(this window's program latency, this window's mvm
  /// latency) instead of their sum, the idealized assumption that whichever
  /// is smaller is fully hidden behind the larger. Carries the same status
  /// as `cim.n_inference_optimum`'s emitted-vs-optimal comparison (spec
  /// docs/roadmap.md M3): a number to compare against, not a replacement
  /// for the real one. Left at 0.0 when doubleBufferCapable is false --
  /// a target that cannot double-buffer has no such projection to make,
  /// the same "leave it at zero rather than guess" discipline
  /// requantizes/reducePartialAdds already follow above for an engine that
  /// does not model something.
  double steadyStateElapsedNsPerInferenceIfOverlapped = 0.0;

  /// True when the target keeps weights across power cycles, so
  /// `installEnergyPj` is genuinely a one-time cost rather than a
  /// per-power-on cost.
  bool persistent = false;

  /// Copied from the target file, for `amortizedInstallEnergyPjPerInference`
  /// below: a volatile array must stay continuously powered to retain its
  /// programmed weights, which non-volatile storage does not need.
  double standbyLeakageUwPerTile = 0.0;
  uint32_t numTiles = 0;
};

/// `stepsPerInference` is how many entries of the use sequence make up one
/// inference; the sequence is expected to be an integral number of those.
/// Pass 0 (or the full sequence length) to treat the whole sequence as a
/// single inference.
CostReport computeCostReport(const TargetSpec &spec,
                              const PlacementResult &result,
                              size_t stepsPerInference);

/// Install energy charged to each inference when the install cost is spread
/// over `inferences` runs, plus -- on a volatile target only -- the standby
/// leakage cost of keeping the array continuously powered so it retains
/// those weights for the whole run (`report.standbyLeakageUwPerTile *
/// report.numTiles`, converted through the run's own per-inference elapsed
/// time). On a non-volatile target (`report.persistent`) this collapses
/// toward zero as `inferences` grows -- amortizing install cost over more
/// runs is what makes weight-stationary CIM worth compiling for. On a
/// volatile target it does not: the leakage term scales with elapsed time,
/// not with the install event, so it is a constant per-inference floor no
/// amount of amortization can shrink -- persistence changes which curve you
/// are on, not just where a fixed curve sits (`docs/roadmap.md`'s M3
/// section, `bench/plots/plot_amortization.py`).
double amortizedInstallEnergyPjPerInference(const CostReport &report,
                                             uint64_t inferences);

/// Same computation as amortizedInstallEnergyPjPerInference, but using
/// steadyStateElapsedNsPerInferenceIfOverlapped for the volatile leakage
/// term instead of steadyStateElapsedNsPerInference -- "what would the
/// amortized cost be if this target's double_buffer_program capability
/// were actually exploited". Identical to
/// amortizedInstallEnergyPjPerInference when `!report.doubleBufferCapable`
/// (there is then no overlapped elapsed time to differ), so it is always
/// safe to call and never reports a number a target without the capability
/// could not, in principle, reach.
double amortizedInstallEnergyPjPerInferenceIfOverlapped(
    const CostReport &report, uint64_t inferences);

/// Render the report as the table from spec Sec. 17.
std::string formatCostReport(const CostReport &report, const std::string &label);

/// Render the report as JSON, the format `cim-cost-report` emits.
std::string toJson(const CostReport &report, const std::string &label);

} // namespace cim

#endif // CIM_PLACEMENT_COSTREPORT_H
