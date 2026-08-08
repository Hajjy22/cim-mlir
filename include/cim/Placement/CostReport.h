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
  double steadyStateElapsedNsPerInference = 0.0;

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

/// Render the report as the table from spec Sec. 17.
std::string formatCostReport(const CostReport &report, const std::string &label);

/// Render the report as JSON, the format `cim-cost-report` emits.
std::string toJson(const CostReport &report, const std::string &label);

} // namespace cim

#endif // CIM_PLACEMENT_COSTREPORT_H
