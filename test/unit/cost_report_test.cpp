//===- cost_report_test.cpp - Cost report formatting -----------*- C++ -*-===//
//
// formatCostReport and toJson were called by nothing in the repository and
// asserted on by nothing. toJson is the format cim-cost-report (spec Sec. 6,
// Pass 8) will emit and that bench/plots reads, so pinning it now means the
// pass is written against a format that already has tests, rather than the
// format being whatever the pass happens to produce.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Placement/CostReport.h"
#include "cim/Placement/Placement.h"
#include "cim/Placement/Workloads.h"

#include <algorithm>
#include <string>

using namespace cim;

namespace {

/// erbium-8t's cost table, built programmatically (these tests must not
/// depend on the filesystem).
TargetSpec erbium8t() {
  TargetSpec spec;
  spec.name = "erbium-8t";
  spec.targetClass = TargetClass::NearMemory;
  spec.provenance = Provenance::Estimated;
  spec.tiles.count = 8;
  spec.tiles.rows = 256;
  spec.tiles.cols = 256;
  spec.tiles.persistent = true;
  spec.tiles.persistence = "nonvolatile";
  spec.costs.program.latencyNs = 12000;
  spec.costs.program.energyPj = 480000;
  spec.costs.mvm.latencyNs = 640;
  spec.costs.mvm.energyPj = 3100;
  spec.costs.transfer.bandwidthGbps = 12.8;
  spec.costs.transfer.energyPjPerByte = 4.2;
  spec.precision.outputEffectiveBits = 8;
  return spec;
}

CostReport fitReport(uint32_t inferences) {
  const Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, inferences);
  const PlacementResult r =
      computePlacement(wl.problem, EvictionPolicy::Belady);
  return computeCostReport(erbium8t(), r, wl.stepsPerInference);
}

/// 16 distinct blocks over 2 tiles: nothing stays resident across an
/// inference's own sweep, so Belady reprograms on nearly every step in
/// steady state -- unlike fitReport's mm-fit, this gives the overlap
/// projection a genuinely nonzero program-vs-mvm mix to take a max() over.
CostReport spillReport(uint32_t inferences, bool doubleBufferCapable) {
  TargetSpec spec = erbium8t();
  spec.tiles.count = 2;
  spec.capabilities.doubleBufferProgram = doubleBufferCapable;
  const Workload wl = makeMatmulWorkload("mm-spill", "", 16, 2, inferences);
  const PlacementResult r =
      computePlacement(wl.problem, EvictionPolicy::Belady);
  return computeCostReport(spec, r, wl.stepsPerInference);
}

} // namespace

CIM_TEST(cost_report_json_has_the_fields_consumers_read) {
  const std::string json = toJson(fitReport(1000), "mm-fit");

  // bench/plots/plot_residency.py reads these by name; renaming one without
  // updating the plot script would break silently at plot time.
  for (const char *field :
       {"\"label\"", "\"persistent\"", "\"programs\"", "\"mvms\"",
        "\"requantizes\"", "\"reuses\"", "\"install_programs\"",
        "\"install_energy_pj\"", "\"install_latency_ns\"",
        "\"steady_state_programs_per_inference\"", "\"total_energy_pj\"",
        "\"total_latency_ns\""})
    CIM_EXPECT_CONTAINS(json, field);

  CIM_EXPECT_CONTAINS(json, "\"label\": \"mm-fit\"");
  CIM_EXPECT_CONTAINS(json, "\"persistent\": true");
  // The headline: a model that fits reprograms nothing per inference.
  CIM_EXPECT_CONTAINS(json, "\"steady_state_programs_per_inference\": 0");
  // Balanced braces -- a malformed JSON blob would break every consumer.
  CIM_EXPECT_EQ(json.find('{'), size_t(0));
  CIM_EXPECT_NE(json.rfind('}'), std::string::npos);
}

CIM_TEST(cost_report_json_reports_a_volatile_target_honestly) {
  TargetSpec volatileSpec = erbium8t();
  volatileSpec.tiles.persistent = false;
  volatileSpec.tiles.persistence = "volatile";

  const Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 10);
  const PlacementResult r =
      computePlacement(wl.problem, EvictionPolicy::Belady);
  const std::string json =
      toJson(computeCostReport(volatileSpec, r, wl.stepsPerInference), "vol");

  // On a volatile target the install cost recurs on every power-on, so the
  // report must not claim persistence the hardware does not have.
  CIM_EXPECT_CONTAINS(json, "\"persistent\": false");
}

CIM_TEST(cost_report_text_shows_install_and_amortization) {
  const std::string text = formatCostReport(fitReport(1000), "mm-fit");

  CIM_EXPECT_CONTAINS(text, "mm-fit");
  CIM_EXPECT_CONTAINS(text, "install:");
  CIM_EXPECT_CONTAINS(text, "inference:");
  CIM_EXPECT_CONTAINS(text, "reuses:");
  // Non-volatile install cost is paid once; the label must say so, since
  // that distinction is the entire argument for compiling to CIM.
  CIM_EXPECT_CONTAINS(text, "once, nonvolatile");
  CIM_EXPECT_CONTAINS(text, "amortized over 1000000 inferences");
}

CIM_TEST(cost_report_text_labels_a_volatile_target_differently) {
  TargetSpec volatileSpec = erbium8t();
  volatileSpec.tiles.persistent = false;

  const Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 10);
  const PlacementResult r =
      computePlacement(wl.problem, EvictionPolicy::Belady);
  const std::string text = formatCostReport(
      computeCostReport(volatileSpec, r, wl.stepsPerInference), "vol");

  CIM_EXPECT_CONTAINS(text, "per power-on, volatile");
}

CIM_TEST(compute_cost_report_leaves_requantize_at_zero) {
  // computeCostReport (this engine) models weight-programming amortization
  // from a PlacementResult alone, which has no notion of a cim.requantize
  // step -- cim-cost-report (lib/Transforms/CIMCostReport.cpp), which walks
  // real compiled IR, is what actually charges for it. Pinning the zero
  // here documents that boundary rather than leaving it implicit.
  const CostReport report = fitReport(1000);
  CIM_EXPECT_EQ(report.requantizes, uint64_t(0));
  CIM_EXPECT(report.requantizeEnergyPj == 0.0);
  CIM_EXPECT(report.requantizeLatencyNs == 0.0);
}

CIM_TEST(compute_cost_report_leaves_reduce_partial_at_zero) {
  // Same boundary as requantize immediately above, for cim.reduce_partial's
  // chained cimrt_reduce_add calls: this engine has no notion of a reduce
  // step either, so cim-cost-report is what actually charges for it.
  const CostReport report = fitReport(1000);
  CIM_EXPECT_EQ(report.reducePartialAdds, uint64_t(0));
  CIM_EXPECT(report.reducePartialEnergyPj == 0.0);
  CIM_EXPECT(report.reducePartialLatencyNs == 0.0);
}

CIM_TEST(compute_cost_report_leaves_the_overlap_projection_at_zero_without_the_capability) {
  // A target that never declares capabilities.double_buffer_program has no
  // overlap projection to make -- left at zero rather than guessed, the
  // same discipline requantize/reduce_partial follow above for an engine
  // that does not model something. fitReport's erbium8t() fixture never
  // sets the flag, so this also exercises the ordinary default.
  const CostReport report = fitReport(1000);
  CIM_EXPECT(!report.doubleBufferCapable);
  CIM_EXPECT(report.steadyStateElapsedNsPerInferenceIfOverlapped == 0.0);
  // ...and the amortization sibling must fall back to the honest,
  // non-projected figure exactly, not merely something close to it.
  CIM_EXPECT_EQ(amortizedInstallEnergyPjPerInferenceIfOverlapped(report, 1000),
               amortizedInstallEnergyPjPerInference(report, 1000));
}

CIM_TEST(cost_report_overlap_projection_is_the_max_and_never_exceeds_the_serial_sum) {
  // mm-spill over 2 tiles: steady state reprograms almost every step, so
  // the window has a real, nonzero mix of program and mvm latency -- unlike
  // fitReport's mm-fit, which has nothing to take a max() over. The claim
  // being tested is exactly what the field's own comment promises: the
  // projection is max(program, mvm), and max is never greater than sum for
  // two non-negative numbers, so it must never exceed the real, serial
  // figure -- overlap can only ever help or do nothing, never hurt.
  const CostReport report = spillReport(1000, /*doubleBufferCapable=*/true);
  CIM_EXPECT(report.doubleBufferCapable);
  CIM_EXPECT(report.steadyStateLatencyNsPerInference > 0.0);

  const double mvmLatencyPerInferenceNs =
      report.steadyStateElapsedNsPerInference -
      report.steadyStateLatencyNsPerInference;
  const double expectedMax = std::max(report.steadyStateLatencyNsPerInference,
                                      mvmLatencyPerInferenceNs);
  CIM_EXPECT(report.steadyStateElapsedNsPerInferenceIfOverlapped == expectedMax);
  CIM_EXPECT(report.steadyStateElapsedNsPerInferenceIfOverlapped <=
             report.steadyStateElapsedNsPerInference);
  // This workload's program cost dominates its mvm cost heavily enough
  // (12000 ns vs. 640 ns, reprogramming on nearly every step) that the
  // projection must be a REAL improvement, not a coincidental tie -- a
  // bug that always returned the serial sum would pass the <= check above
  // and still be wrong.
  CIM_EXPECT(report.steadyStateElapsedNsPerInferenceIfOverlapped <
             report.steadyStateElapsedNsPerInference);
}

CIM_TEST(amortization_if_overlapped_never_exceeds_the_serial_figure_on_a_volatile_target) {
  // The amortization-level version of the same "overlap only ever helps"
  // claim, on a volatile leaky target where the elapsed-time figure
  // actually feeds a real number (the leakage floor) rather than being
  // inert.
  TargetSpec spec = erbium8t();
  spec.tiles.count = 2;
  spec.tiles.persistent = false;
  spec.tiles.persistence = "volatile";
  spec.costs.standbyLeakageUwPerTile = 4.5;
  spec.capabilities.doubleBufferProgram = true;
  const Workload wl = makeMatmulWorkload("mm-spill-volatile", "", 16, 2, 1000);
  const PlacementResult r =
      computePlacement(wl.problem, EvictionPolicy::Belady);
  const CostReport report = computeCostReport(spec, r, wl.stepsPerInference);

  const double serial = amortizedInstallEnergyPjPerInference(report, 1000000);
  const double overlapped =
      amortizedInstallEnergyPjPerInferenceIfOverlapped(report, 1000000);
  CIM_EXPECT(overlapped <= serial);
  CIM_EXPECT(overlapped < serial); // a real improvement, not a coincidence
}

CIM_TEST(cost_report_json_and_text_surface_the_overlap_projection_honestly) {
  const CostReport capable = spillReport(1000, /*doubleBufferCapable=*/true);
  const CostReport notCapable = spillReport(1000, /*doubleBufferCapable=*/false);

  const std::string jsonCapable = toJson(capable, "spill-dbuf");
  CIM_EXPECT_CONTAINS(jsonCapable, "\"double_buffer_capable\": true");
  CIM_EXPECT_CONTAINS(jsonCapable,
                      "\"steady_state_elapsed_ns_per_inference_if_overlapped\"");

  const std::string jsonNotCapable = toJson(notCapable, "spill-nodbuf");
  CIM_EXPECT_CONTAINS(jsonNotCapable, "\"double_buffer_capable\": false");
  // The key is always present (fixed schema, matching requantizes/
  // reduce_partial_adds above) even when the value is the zero default.
  CIM_EXPECT_CONTAINS(
      jsonNotCapable,
      "\"steady_state_elapsed_ns_per_inference_if_overlapped\": 0");

  // Text only shows the projection line for a target that could ever
  // realize it -- showing it unconditionally would imply every target can
  // double-buffer, which is exactly the "plausible but not achievable"
  // number this project's discipline refuses to print.
  CIM_EXPECT_CONTAINS(formatCostReport(capable, "spill-dbuf"),
                      "if double-buffered");
  CIM_EXPECT(formatCostReport(notCapable, "spill-nodbuf")
                 .find("if double-buffered") == std::string::npos);
}

CIM_TEST(cost_report_energy_units_scale_sensibly) {
  // The formatter picks a unit by magnitude. Getting this wrong is how a
  // plot ends up off by a factor of a thousand -- exactly the discrepancy
  // documented in docs/target-format.md.
  const std::string text = formatCostReport(fitReport(1000), "units");
  // 8 programs x 480000 pJ = 3.84e6 pJ, which should read as microjoules.
  CIM_EXPECT_CONTAINS(text, "uJ");
}

CIM_TEST(action_kind_and_policy_names_round_trip) {
  // Used in cim-bench's JSON and in diagnostics; an unnamed enum value
  // would surface as "unknown" in a results file.
  CIM_EXPECT_EQ(std::string(toString(EvictionPolicy::Belady)), "belady");
  CIM_EXPECT_EQ(std::string(toString(EvictionPolicy::LRU)), "lru");
  CIM_EXPECT_EQ(std::string(toString(EvictionPolicy::FIFO)), "fifo");

  CIM_EXPECT_EQ(std::string(toString(ActionKind::Reuse)), "reuse");
  CIM_EXPECT_EQ(std::string(toString(ActionKind::ProgramIntoFree)), "program");
  CIM_EXPECT_EQ(std::string(toString(ActionKind::ProgramWithEviction)),
                "program+evict");
}

CIM_TEST(amortization_matches_the_old_formula_when_leakage_is_zero) {
  // erbium-8t.yaml itself declares standby_leakage_uw_per_tile: 0.0 even
  // though it is nonvolatile -- backward-compatibility case: a volatile
  // target that happens to declare zero leakage must amortize exactly like
  // the pre-leakage formula (plain division), not acquire a floor from
  // nowhere.
  TargetSpec volatileNoLeakage = erbium8t();
  volatileNoLeakage.tiles.persistent = false;
  volatileNoLeakage.tiles.persistence = "volatile";
  volatileNoLeakage.costs.standbyLeakageUwPerTile = 0.0;

  const Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 1000);
  const PlacementResult r = computePlacement(wl.problem, EvictionPolicy::Belady);
  const CostReport report =
      computeCostReport(volatileNoLeakage, r, wl.stepsPerInference);

  const double over1M = amortizedInstallEnergyPjPerInference(report, 1000000);
  const double over100M =
      amortizedInstallEnergyPjPerInference(report, 100000000);
  CIM_EXPECT(over100M < over1M);
  CIM_EXPECT_EQ(over100M, report.installEnergyPj / 100000000.0);
}

CIM_TEST(amortization_has_a_nonzero_floor_on_a_volatile_target_with_leakage) {
  // The actual roadmap claim (docs/roadmap.md's M3 section): persistence
  // changes which curve the amortized cost is on, not just where a single
  // curve sits. Two reports built from the identical placement result and
  // install cost, differing only in persistence and leakage -- so any
  // difference below is provably caused by that, and nothing else.
  TargetSpec volatileSpec = erbium8t();
  volatileSpec.tiles.persistent = false;
  volatileSpec.tiles.persistence = "volatile";
  volatileSpec.costs.standbyLeakageUwPerTile = 4.5; // generic-digital-cim.yaml

  const Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 1000);
  const PlacementResult r = computePlacement(wl.problem, EvictionPolicy::Belady);
  const CostReport nonvolatileReport = computeCostReport(erbium8t(), r, wl.stepsPerInference);
  const CostReport volatileReport = computeCostReport(volatileSpec, r, wl.stepsPerInference);

  const double floorPj = static_cast<double>(volatileReport.numTiles) *
                         volatileReport.standbyLeakageUwPerTile *
                         volatileReport.steadyStateElapsedNsPerInference * 1e-3;
  CIM_EXPECT(floorPj > 0.0);

  // Amortizing over ever more inferences keeps shrinking the non-volatile
  // number toward zero...
  const double nonvolatile1M =
      amortizedInstallEnergyPjPerInference(nonvolatileReport, 1000000);
  const double nonvolatile100M =
      amortizedInstallEnergyPjPerInference(nonvolatileReport, 100000000);
  CIM_EXPECT(nonvolatile100M < nonvolatile1M);
  CIM_EXPECT(nonvolatile100M < floorPj);

  // ...but the volatile number never drops below the leakage floor, no
  // matter how many more inferences it is amortized over -- the qualitative
  // difference, not merely a different number at the same N.
  const double volatile1M =
      amortizedInstallEnergyPjPerInference(volatileReport, 1000000);
  const double volatile100M =
      amortizedInstallEnergyPjPerInference(volatileReport, 100000000);
  CIM_EXPECT(volatile100M >= floorPj);
  CIM_EXPECT(volatile1M >= floorPj);
  // At a large enough N the install term is negligible and the reported
  // value converges to (not just approaches) the floor.
  const double volatileHugeN =
      amortizedInstallEnergyPjPerInference(volatileReport, 1000000000000ULL);
  CIM_EXPECT(volatileHugeN - floorPj < floorPj * 1e-6 + 1e-9);

  // At the same N, the volatile number must be strictly worse -- this is
  // the number a plot of both curves actually shows.
  CIM_EXPECT(volatile100M > nonvolatile100M);
}

CIM_TEST(amortization_is_monotone_and_guards_division_by_zero) {
  const CostReport report = fitReport(1000);
  const double over1 = amortizedInstallEnergyPjPerInference(report, 1);
  const double over1k = amortizedInstallEnergyPjPerInference(report, 1000);
  const double over1M = amortizedInstallEnergyPjPerInference(report, 1000000);

  CIM_EXPECT(over1 > over1k);
  CIM_EXPECT(over1k > over1M);
  CIM_EXPECT(over1M > 0.0);
  // Zero inferences must not produce inf or NaN in a report.
  CIM_EXPECT_EQ(amortizedInstallEnergyPjPerInference(report, 0), 0.0);
}

CIM_TEST(cost_report_handles_an_empty_placement) {
  // cim-cost-report will run on modules with no cim ops at all.
  PlacementProblem empty;
  empty.numTiles = 8;
  const PlacementResult r = computePlacement(empty, EvictionPolicy::Belady);
  const CostReport report = computeCostReport(erbium8t(), r, 0);

  CIM_EXPECT_EQ(report.programs, 0u);
  CIM_EXPECT_EQ(report.mvms, 0u);
  CIM_EXPECT_EQ(report.installPrograms, 0u);
  // Must still emit well-formed JSON rather than an empty string.
  CIM_EXPECT_CONTAINS(toJson(report, "empty"), "\"programs\": 0");
}
