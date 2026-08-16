//===- cost_model.h - Analytical cost model (spec Sec. 9.2) ----*- C++ -*-===//
//
// Deliberately NOT cycle-accurate: sums target-file-declared costs over the
// executed trace rather than simulating timing. Cycle-accurate simulation
// is a multi-year project; the interesting result (program-vs-compute
// asymmetry, residency wins from cim-placement) is visible analytically.
//
// TODO(spec Sec.9.2): if real timing is later needed, integrate Ramulator2
// (SAFARI group) rather than rebuilding a cycle-accurate model here.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_RUNTIME_SIMULATOR_COST_MODEL_H
#define CIM_RUNTIME_SIMULATOR_COST_MODEL_H

#include "cim/Target/TargetSpec.h"
#include <cstdint>

namespace cim {

/// Accumulates cost over a trace of program/mvm/transfer events, using a
/// single TargetSpec's cost table. Backs both cimrt's profiling counters
/// (runtime/include/cimrt.h) and the cim-cost-report pass (spec Sec. 6,
/// Pass 8) when running against the functional simulator.
class CostAccumulator {
public:
  /// A point-in-time copy of the counters, cheap enough to take on every
  /// cimrt_profile_start without worrying about the cost of profiling
  /// itself. Backs windowed (start/stop-delta) profiling in cimrt_profile.
  struct Snapshot {
    uint64_t programsIssued = 0;
    uint64_t mvmsIssued = 0;
    uint64_t requantizesIssued = 0;
    uint64_t reduceAddsIssued = 0;
    uint64_t bytesTransferred = 0;
    double energyPj = 0.0;
    double latencyNs = 0.0;
  };

  explicit CostAccumulator(const TargetSpec &targetSpec) : spec(targetSpec) {}

  Snapshot snapshot() const {
    return Snapshot{programsIssued,   mvmsIssued, requantizesIssued,
                    reduceAddsIssued, bytesTransferred, energyPj,
                    latencyNs};
  }

  void recordProgram() {
    ++programsIssued;
    latencyNs += spec.costs.program.latencyNs;
    energyPj += spec.costs.program.energyPj;
  }

  void recordMvm() {
    ++mvmsIssued;
    latencyNs += spec.costs.mvm.latencyNs;
    energyPj += spec.costs.mvm.energyPj;
  }

  void recordRequantize() {
    ++requantizesIssued;
    latencyNs += spec.costs.requantize.latencyNs;
    energyPj += spec.costs.requantize.energyPj;
  }

  /// One cimrt_reduce_add call -- NOT one cim.reduce_partial op. An
  /// N-operand cim.reduce_partial lowers to N-1 of these (lowerReducePartial,
  /// lib/Transforms/CIMLowerToTarget.cpp), and cim-cost-report weights its
  /// static side the same way (CostReportUtils.cpp), so the two sides of
  /// test/mlir/cost_report_e2e_test.cpp's differential count the same events.
  void recordReduceAdd() {
    ++reduceAddsIssued;
    latencyNs += spec.costs.reducePartial.latencyNs;
    energyPj += spec.costs.reducePartial.energyPj;
  }

  void recordTransfer(uint64_t bytes) {
    bytesTransferred += bytes;
    energyPj += static_cast<double>(bytes) * spec.costs.transfer.energyPjPerByte;

    // THE UNITS, PINNED. `bandwidth_gbps` is read as GIGABYTES per second,
    // not gigabits -- despite the field name, which is inherited from the
    // spec and is misleading. Two independent reasons, because an 8x
    // silent error in every latency number is exactly the class of
    // plausible-but-wrong this project refuses:
    //
    //   1. The shipped values only make sense as GB/s: erbium-8t 12.8,
    //      generic-digital-cim 25.6, upmem-like 6.4 -- textbook DDR
    //      channel bandwidths in GB/s. As gigabits those would describe
    //      implausibly slow memory for the devices they model.
    //   2. Reading them as GB/s makes the conversion exact and unit-free:
    //      1 GB/s IS 1 byte per nanosecond, so ns = bytes / gbps with no
    //      scale factor to get backwards. (A bits reading would need a
    //      /8 that nothing in the schema mentions.)
    //
    // Same discipline as install_energy_pins_the_pj_as_written_convention
    // in test/unit/cost_report_test.cpp: the convention is pinned by a
    // test, not just asserted in a comment, so a future "fix" cannot
    // silently rescale every published latency. Guarded against a zero or
    // negative bandwidth, which the parser already rejects, so this is
    // belt-and-braces rather than a live path.
    if (spec.costs.transfer.bandwidthGbps > 0.0)
      latencyNs +=
          static_cast<double>(bytes) / spec.costs.transfer.bandwidthGbps;
  }

  uint64_t getProgramsIssued() const { return programsIssued; }
  uint64_t getMvmsIssued() const { return mvmsIssued; }
  uint64_t getRequantizesIssued() const { return requantizesIssued; }
  uint64_t getReduceAddsIssued() const { return reduceAddsIssued; }
  uint64_t getBytesTransferred() const { return bytesTransferred; }
  double getEstimatedEnergyPj() const { return energyPj; }
  double getEstimatedLatencyNs() const { return latencyNs; }

private:
  const TargetSpec &spec;
  uint64_t programsIssued = 0;
  uint64_t mvmsIssued = 0;
  uint64_t requantizesIssued = 0;
  uint64_t reduceAddsIssued = 0;
  uint64_t bytesTransferred = 0;
  double energyPj = 0.0;
  double latencyNs = 0.0;
};

} // namespace cim

#endif // CIM_RUNTIME_SIMULATOR_COST_MODEL_H
