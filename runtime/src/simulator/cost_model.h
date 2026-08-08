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
    uint64_t bytesTransferred = 0;
    double energyPj = 0.0;
    double latencyNs = 0.0;
  };

  explicit CostAccumulator(const TargetSpec &targetSpec) : spec(targetSpec) {}

  Snapshot snapshot() const {
    return Snapshot{programsIssued, mvmsIssued, bytesTransferred, energyPj,
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

  void recordTransfer(uint64_t bytes) {
    bytesTransferred += bytes;
    energyPj += static_cast<double>(bytes) * spec.costs.transfer.energyPjPerByte;
    // TODO(spec Sec.9.2): fold in bandwidth_gbps to derive transfer latency.
  }

  uint64_t getProgramsIssued() const { return programsIssued; }
  uint64_t getMvmsIssued() const { return mvmsIssued; }
  uint64_t getBytesTransferred() const { return bytesTransferred; }
  double getEstimatedEnergyPj() const { return energyPj; }
  double getEstimatedLatencyNs() const { return latencyNs; }

private:
  const TargetSpec &spec;
  uint64_t programsIssued = 0;
  uint64_t mvmsIssued = 0;
  uint64_t bytesTransferred = 0;
  double energyPj = 0.0;
  double latencyNs = 0.0;
};

} // namespace cim

#endif // CIM_RUNTIME_SIMULATOR_COST_MODEL_H
