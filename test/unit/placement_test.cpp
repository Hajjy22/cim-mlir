//===- placement_test.cpp - Tests for the placement engine -----*- C++ -*-===//
//
// Every performance claim here has a correctness check next to it: each
// schedule is replayed through validatePlacement() before its program count
// is trusted (spec Sec. 10).
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Placement/CostReport.h"
#include "cim/Placement/Placement.h"
#include "cim/Placement/Workloads.h"

using namespace cim;

namespace {

/// Run a policy and assert the resulting schedule is internally valid.
PlacementResult runValidated(const PlacementProblem &problem,
                              EvictionPolicy policy) {
  PlacementResult result = computePlacement(problem, policy);
  std::string error;
  if (!validatePlacement(problem, result, &error))
    ::cimtest::reportFailure(__FILE__, __LINE__,
                             std::string("invalid schedule (") +
                                 toString(policy) + "): " + error);
  return result;
}

/// The erbium-8t target, transcribed from targets/erbium-8t.yaml. Built
/// programmatically so these tests need no YAML parser (that path goes
/// through llvm::yaml and therefore needs an LLVM toolchain).
TargetSpec erbium8t() {
  TargetSpec spec;
  spec.name = "erbium-8t";
  spec.targetClass = TargetClass::NearMemory;
  spec.provenance = Provenance::Estimated;
  spec.tiles.count = 8;
  spec.tiles.rows = 256;
  spec.tiles.cols = 256;
  spec.tiles.weightDtype = "i8";
  spec.tiles.activationDtype = "i8";
  spec.tiles.accumulatorDtype = "i32";
  spec.tiles.persistent = true;
  spec.tiles.persistence = "nonvolatile";
  spec.costs.program.latencyNs = 12000;
  spec.costs.program.energyPj = 480000;
  spec.costs.mvm.latencyNs = 640;
  spec.costs.mvm.energyPj = 3100;
  spec.costs.transfer.bandwidthGbps = 12.8;
  spec.costs.transfer.energyPjPerByte = 4.2;
  spec.costs.standbyLeakageUwPerTile = 0.0;
  spec.precision.outputEffectiveBits = 8;
  return spec;
}

} // namespace

//===----------------------------------------------------------------------===//
// Partitioning
//===----------------------------------------------------------------------===//

CIM_TEST(partition_matches_spec_worked_example) {
  // Spec Sec. 17: a [512 x 1024] INT8 matmul on 256x256 tiles partitions
  // into ceil(512/256) = 2 in K and ceil(1024/256) = 4 in N -> 8 sub-tiles.
  CIM_EXPECT_EQ(partitionBlockCount(512, 1024, 256, 256), 8u);
}

CIM_TEST(partition_zero_pads_ragged_edges) {
  // A ragged edge counts as a whole block, because cim-partition zero-pads
  // rather than emitting a special-cased tile shape.
  CIM_EXPECT_EQ(partitionBlockCount(257, 256, 256, 256), 2u);
  CIM_EXPECT_EQ(partitionBlockCount(1, 1, 256, 256), 1u);
}

//===----------------------------------------------------------------------===//
// The headline result: a model that fits reprograms nothing after install
//===----------------------------------------------------------------------===//

CIM_TEST(fitting_model_programs_each_weight_exactly_once) {
  // 8 weight blocks, 8 tiles, 50 inferences.
  Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 50);
  PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);

  CIM_EXPECT_EQ(r.programs, 8u);              // install only
  CIM_EXPECT_EQ(r.distinctWeights, 8u);
  CIM_EXPECT_EQ(r.evictions, 0u);
  CIM_EXPECT_EQ(r.reuses, 8u * 50u - 8u);     // every later use is free
}

CIM_TEST(fitting_model_costs_nothing_in_steady_state) {
  Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 50);
  PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);
  CostReport report = computeCostReport(erbium8t(), r, wl.stepsPerInference);

  // Install is 8 programs; steady-state inference is zero. This is the
  // whole argument for compiling to non-volatile CIM.
  CIM_EXPECT_EQ(report.installPrograms, 8u);
  CIM_EXPECT_EQ(report.steadyStateProgramsPerInference, 0u);
  CIM_EXPECT(report.persistent);

  // Spec Sec. 17 timings: 8 x 12000 ns = 96 us install, 8 x 640 ns per
  // inference of MVM.
  CIM_EXPECT(report.installLatencyNs == 96000.0);
  CIM_EXPECT(report.mvmLatencyNs == 8.0 * 50.0 * 640.0);
}

CIM_TEST(all_policies_agree_when_the_model_fits) {
  // With no eviction pressure there is nothing to be clever about, so a
  // weaker policy must not look better or worse.
  Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 10);
  const uint64_t belady = runValidated(wl.problem, EvictionPolicy::Belady).programs;
  const uint64_t lru = runValidated(wl.problem, EvictionPolicy::LRU).programs;
  const uint64_t fifo = runValidated(wl.problem, EvictionPolicy::FIFO).programs;
  CIM_EXPECT_EQ(belady, 8u);
  CIM_EXPECT_EQ(lru, 8u);
  CIM_EXPECT_EQ(fifo, 8u);
}

//===----------------------------------------------------------------------===//
// Under spill pressure, compile-time knowledge beats a runtime cache
//===----------------------------------------------------------------------===//

CIM_TEST(belady_is_never_worse_than_lru_or_fifo) {
  for (uint32_t blocks : {9u, 12u, 16u, 33u, 64u}) {
    Workload wl = makeMatmulWorkload("spill", "", blocks, 8, 12);
    const uint64_t belady =
        runValidated(wl.problem, EvictionPolicy::Belady).programs;
    const uint64_t lru = runValidated(wl.problem, EvictionPolicy::LRU).programs;
    const uint64_t fifo = runValidated(wl.problem, EvictionPolicy::FIFO).programs;
    CIM_EXPECT_LE(belady, lru);
    CIM_EXPECT_LE(belady, fifo);
  }
}

CIM_TEST(belady_beats_lru_on_the_cyclic_sweep) {
  // A cyclic sweep over more blocks than tiles is the pathological case for
  // LRU: it evicts exactly the block it is about to need. Belady evicts the
  // furthest-in-future instead and keeps most of its working set. This is
  // the mm-spill-2x row of the benchmark table, and it must show a real
  // gap, not just a tie.
  Workload wl = makeMatmulWorkload("mm-spill-2x", "", 16, 8, 12);
  const uint64_t belady = runValidated(wl.problem, EvictionPolicy::Belady).programs;
  const uint64_t lru = runValidated(wl.problem, EvictionPolicy::LRU).programs;

  CIM_EXPECT(belady < lru);
  // LRU thrashes completely: every one of the 16*12 uses misses.
  CIM_EXPECT_EQ(lru, 16u * 12u);
}

CIM_TEST(spill_still_programs_at_least_every_distinct_weight_once) {
  // Sanity floor: you cannot use a weight you never programmed.
  for (uint32_t blocks : {9u, 16u, 64u}) {
    Workload wl = makeMatmulWorkload("spill", "", blocks, 8, 5);
    PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);
    CIM_EXPECT_LE(static_cast<uint64_t>(blocks), r.programs);
    CIM_EXPECT_EQ(r.distinctWeights, static_cast<uint64_t>(blocks));
  }
}

//===----------------------------------------------------------------------===//
// Cross-layer scheduling
//===----------------------------------------------------------------------===//

CIM_TEST(layered_model_that_fits_installs_once) {
  // 3 layers of 2 blocks each = 6 blocks on 8 tiles: fits.
  Workload wl = makeLayeredWorkload("mlp-3layer", "", {2, 2, 2}, 8, 20);
  PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);
  CIM_EXPECT_EQ(r.programs, 6u);
  CIM_EXPECT_EQ(r.evictions, 0u);
}

CIM_TEST(layers_do_not_share_weights) {
  // Two layers of 4 blocks must produce 8 distinct weights competing for
  // tiles, not 4 aliased ones — otherwise cross-layer pressure is fiction.
  Workload wl = makeLayeredWorkload("two-layer", "", {4, 4}, 8, 1);
  PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);
  CIM_EXPECT_EQ(r.distinctWeights, 8u);
  CIM_EXPECT_EQ(wl.stepsPerInference, size_t(8));
}

//===----------------------------------------------------------------------===//
// Schedule validity
//===----------------------------------------------------------------------===//

CIM_TEST(validator_rejects_a_corrupted_schedule) {
  // The validator is load-bearing — if it rubber-stamps anything, every
  // "validated" result above is meaningless. Corrupt a schedule and check
  // it is actually caught.
  Workload wl = makeMatmulWorkload("mm-fit", "", 4, 4, 3);
  PlacementResult r = computePlacement(wl.problem, EvictionPolicy::Belady);
  std::string error;
  CIM_EXPECT(validatePlacement(wl.problem, r, &error));

  // Point a reuse at the wrong tile.
  PlacementResult corrupted = r;
  for (auto &a : corrupted.actions) {
    if (a.kind == ActionKind::Reuse) {
      a.tile = (a.tile + 1) % wl.problem.numTiles;
      break;
    }
  }
  CIM_EXPECT(!validatePlacement(wl.problem, corrupted, &error));

  // Under-report the program count.
  PlacementResult miscounted = r;
  miscounted.programs -= 1;
  CIM_EXPECT(!validatePlacement(wl.problem, miscounted, &error));
}

CIM_TEST(degenerate_inputs_are_handled) {
  PlacementProblem empty;
  empty.numTiles = 8;
  PlacementResult r = computePlacement(empty, EvictionPolicy::Belady);
  CIM_EXPECT_EQ(r.programs, 0u);
  CIM_EXPECT_EQ(r.actions.size(), size_t(0));

  // Zero tiles is unsatisfiable, and must not loop or crash.
  PlacementProblem noTiles;
  noTiles.numTiles = 0;
  noTiles.useSequence = {0, 1, 2};
  PlacementResult r2 = computePlacement(noTiles, EvictionPolicy::Belady);
  CIM_EXPECT_EQ(r2.actions.size(), size_t(0));
}

//===----------------------------------------------------------------------===//
// Cost model
//===----------------------------------------------------------------------===//

CIM_TEST(install_cost_amortizes_toward_zero) {
  Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 100);
  PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);
  CostReport report = computeCostReport(erbium8t(), r, wl.stepsPerInference);

  // 8 programs x 480000 pJ = 3.84e6 pJ of install energy.
  CIM_EXPECT(report.installEnergyPj == 8.0 * 480000.0);

  const double over1 = amortizedInstallEnergyPjPerInference(report, 1);
  const double over1M = amortizedInstallEnergyPjPerInference(report, 1000000);
  const double over100M = amortizedInstallEnergyPjPerInference(report, 100000000);
  CIM_EXPECT(over1 > over1M);
  CIM_EXPECT(over1M > over100M);
  CIM_EXPECT(over1M == (8.0 * 480000.0) / 1e6);

  // Guard against a divide-by-zero rather than a silent inf.
  CIM_EXPECT(amortizedInstallEnergyPjPerInference(report, 0) == 0.0);
}

CIM_TEST(volatile_target_is_reported_as_non_persistent) {
  TargetSpec volatileSpec = erbium8t();
  volatileSpec.tiles.persistent = false;
  volatileSpec.tiles.persistence = "volatile";

  Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 10);
  PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);
  CostReport report = computeCostReport(volatileSpec, r, wl.stepsPerInference);

  // Placement is identical; what changes is whether install is a one-time
  // cost. The report must not quietly claim persistence the target lacks.
  CIM_EXPECT(!report.persistent);
  CIM_EXPECT_EQ(report.installPrograms, 8u);
}

CIM_TEST(mvm_count_covers_every_step_not_just_misses) {
  Workload wl = makeMatmulWorkload("mm-fit", "", 8, 8, 10);
  PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);
  CostReport report = computeCostReport(erbium8t(), r, wl.stepsPerInference);
  // A reused weight still costs an MVM — reuse saves the program, not the
  // compute.
  CIM_EXPECT_EQ(report.mvms, 80u);
}

//===----------------------------------------------------------------------===//
// The v0.1 workload table
//===----------------------------------------------------------------------===//

CIM_TEST(v01_workload_table_is_well_formed) {
  std::vector<Workload> workloads = makeV01Workloads(8, 256, 256, 10);
  CIM_EXPECT_EQ(workloads.size(), size_t(5));

  for (const Workload &wl : workloads) {
    CIM_EXPECT(!wl.problem.useSequence.empty());
    CIM_EXPECT(wl.stepsPerInference > 0);
    // The sequence must be an integral number of inference windows, or the
    // steady-state column of the cost report is meaningless.
    CIM_EXPECT_EQ(wl.problem.useSequence.size() % wl.stepsPerInference, size_t(0));

    PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);
    CIM_EXPECT_LE(r.distinctWeights, r.programs);
  }
}

CIM_TEST(mm_fit_row_matches_the_spec_worked_example) {
  // The mm-fit row is sized to be the spec Sec. 17 example: a [512 x 1024]
  // matmul filling all 8 tiles of a 256x256 device exactly.
  std::vector<Workload> workloads = makeV01Workloads(8, 256, 256, 10);
  const Workload &mmFit = workloads[0];
  CIM_EXPECT_EQ(mmFit.name, std::string("mm-fit"));
  CIM_EXPECT_EQ(mmFit.stepsPerInference, size_t(8));

  PlacementResult r = runValidated(mmFit.problem, EvictionPolicy::Belady);
  CIM_EXPECT_EQ(r.programs, 8u);
  CIM_EXPECT_EQ(r.evictions, 0u);
}

CIM_TEST(spill_rows_actually_spill) {
  std::vector<Workload> workloads = makeV01Workloads(8, 256, 256, 10);
  for (const Workload &wl : workloads) {
    if (wl.name != "mm-spill-2x" && wl.name != "mm-spill-8x")
      continue;
    PlacementResult r = runValidated(wl.problem, EvictionPolicy::Belady);
    // If a "spill" workload fits, the benchmark is not testing what it claims.
    CIM_EXPECT(r.evictions > 0);
    CIM_EXPECT(r.programs > r.distinctWeights);
  }
}
