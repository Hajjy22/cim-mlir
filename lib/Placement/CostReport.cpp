//===- CostReport.cpp - Analytical cost of a placement ---------*- C++ -*-===//

#include "cim/Placement/CostReport.h"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace cim {

namespace {

/// Count programs (and their cost) within [begin, end) of the action list.
struct WindowCost {
  uint64_t programs = 0;
  double energyPj = 0.0;
  double latencyNs = 0.0;
};

WindowCost costOfWindow(const TargetSpec &spec, const PlacementResult &result,
                         size_t begin, size_t end) {
  WindowCost wc;
  if (begin >= result.actions.size())
    return wc;
  end = std::min(end, result.actions.size());
  for (size_t i = begin; i < end; ++i) {
    if (result.actions[i].kind != ActionKind::Reuse)
      ++wc.programs;
  }
  wc.energyPj = static_cast<double>(wc.programs) * spec.costs.program.energyPj;
  wc.latencyNs = static_cast<double>(wc.programs) * spec.costs.program.latencyNs;
  return wc;
}

std::string formatEnergy(double pj) {
  char buf[64];
  if (pj >= 1e9)
    std::snprintf(buf, sizeof(buf), "%.3f mJ", pj / 1e9);
  else if (pj >= 1e6)
    std::snprintf(buf, sizeof(buf), "%.3f uJ", pj / 1e6);
  else if (pj >= 1e3)
    std::snprintf(buf, sizeof(buf), "%.3f nJ", pj / 1e3);
  else
    std::snprintf(buf, sizeof(buf), "%.3f pJ", pj);
  return buf;
}

std::string formatTime(double ns) {
  char buf[64];
  if (ns >= 1e9)
    std::snprintf(buf, sizeof(buf), "%.3f s", ns / 1e9);
  else if (ns >= 1e6)
    std::snprintf(buf, sizeof(buf), "%.3f ms", ns / 1e6);
  else if (ns >= 1e3)
    std::snprintf(buf, sizeof(buf), "%.3f us", ns / 1e3);
  else
    std::snprintf(buf, sizeof(buf), "%.3f ns", ns);
  return buf;
}

} // namespace

CostReport computeCostReport(const TargetSpec &spec,
                              const PlacementResult &result,
                              size_t stepsPerInference) {
  CostReport report;
  report.persistent = spec.tiles.persistent;
  report.standbyLeakageUwPerTile = spec.costs.standbyLeakageUwPerTile;
  report.numTiles = spec.tiles.count;

  report.programs = result.programs;
  report.reuses = result.reuses;
  // Every step of the use sequence is one matrix-vector multiply against
  // the resident tile, whether or not that step needed a reprogram.
  report.mvms = result.actions.size();

  report.programEnergyPj =
      static_cast<double>(report.programs) * spec.costs.program.energyPj;
  report.programLatencyNs =
      static_cast<double>(report.programs) * spec.costs.program.latencyNs;
  report.mvmEnergyPj =
      static_cast<double>(report.mvms) * spec.costs.mvm.energyPj;
  report.mvmLatencyNs =
      static_cast<double>(report.mvms) * spec.costs.mvm.latencyNs;
  report.totalEnergyPj = report.programEnergyPj + report.mvmEnergyPj;
  report.totalLatencyNs = report.programLatencyNs + report.mvmLatencyNs;

  const size_t window = (stepsPerInference == 0 || stepsPerInference > result.actions.size())
                            ? result.actions.size()
                            : stepsPerInference;
  if (window == 0)
    return report;

  const WindowCost install = costOfWindow(spec, result, 0, window);
  report.installPrograms = install.programs;
  report.installEnergyPj = install.energyPj;
  report.installLatencyNs = install.latencyNs;

  // Steady state = the last full window. If only one window was modeled,
  // there is no steady state distinct from install and we report the same
  // window rather than inventing a zero.
  const size_t windows = result.actions.size() / window;
  const size_t lastBegin = (windows > 0) ? (windows - 1) * window : 0;
  const WindowCost steady = costOfWindow(spec, result, lastBegin, lastBegin + window);
  report.steadyStateProgramsPerInference = steady.programs;
  report.steadyStateEnergyPjPerInference = steady.energyPj;
  report.steadyStateLatencyNsPerInference = steady.latencyNs;

  // Every step in the window is one mvm regardless of whether it also
  // reprograms, so this is exact, not an estimate on top of an estimate.
  const double mvmLatencyPerInferenceNs =
      static_cast<double>(window) * spec.costs.mvm.latencyNs;
  report.steadyStateElapsedNsPerInference =
      steady.latencyNs + mvmLatencyPerInferenceNs;

  return report;
}

double amortizedInstallEnergyPjPerInference(const CostReport &report,
                                             uint64_t inferences) {
  if (inferences == 0)
    return 0.0;
  const double amortizedInstall =
      report.installEnergyPj / static_cast<double>(inferences);
  if (report.persistent)
    return amortizedInstall;

  // Volatile: the array must stay continuously powered for the whole run to
  // retain the weights this pass just installed, and that draws
  // standbyLeakageUwPerTile per tile the entire time -- whether the array
  // is reprogramming, computing, or between the two -- not just while a
  // program or mvm is actually executing, which is why this uses the
  // window's total elapsed time rather than either cost alone. That power
  // cost scales with how long the run takes, not with how many inferences
  // share the one-time install event, so unlike amortizedInstall it does
  // not shrink as `inferences` grows -- every inference pays its own slice
  // of it. Energy unit note: 1 uW held for 1 ns is 1e-15 J = 1e-3 pJ,
  // matching the uW-and-ns units the target schema and TargetSpec already
  // use.
  const double leakagePjPerInference =
      static_cast<double>(report.numTiles) * report.standbyLeakageUwPerTile *
      report.steadyStateElapsedNsPerInference * 1e-3;
  return amortizedInstall + leakagePjPerInference;
}

std::string formatCostReport(const CostReport &report,
                              const std::string &label) {
  std::ostringstream os;
  os << label << "\n";
  os << "  install:   " << report.installPrograms << " programs  "
     << formatTime(report.installLatencyNs) << "  "
     << formatEnergy(report.installEnergyPj)
     << (report.persistent ? "   (once, nonvolatile)" : "   (per power-on, volatile)")
     << "\n";
  os << "  inference: " << report.steadyStateProgramsPerInference
     << " programs  " << formatTime(report.steadyStateLatencyNsPerInference)
     << "  " << formatEnergy(report.steadyStateEnergyPjPerInference) << "\n";
  os << "  mvm total: " << report.mvms << " mvms  "
     << formatTime(report.mvmLatencyNs) << "  "
     << formatEnergy(report.mvmEnergyPj) << "\n";
  if (report.requantizes > 0)
    os << "  requantize total: " << report.requantizes << " requantizes  "
       << formatTime(report.requantizeLatencyNs) << "  "
       << formatEnergy(report.requantizeEnergyPj) << "\n";
  os << "  reuses:    " << report.reuses << " (steps needing no reprogram)\n";

  if (!report.persistent && report.standbyLeakageUwPerTile > 0.0) {
    const double floor = static_cast<double>(report.numTiles) *
                         report.standbyLeakageUwPerTile *
                         report.steadyStateElapsedNsPerInference * 1e-3;
    os << "  standby leakage floor: " << formatEnergy(floor)
       << "/inference (this target never amortizes below it)\n";
  }

  for (uint64_t n : {uint64_t(1), uint64_t(1000000), uint64_t(100000000)}) {
    const double perInf = amortizedInstallEnergyPjPerInference(report, n);
    os << "  install amortized over " << n << " inferences: "
       << formatEnergy(perInf) << "/inference\n";
  }
  return os.str();
}

std::string toJson(const CostReport &report, const std::string &label) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"label\": \"" << label << "\",\n";
  os << "  \"persistent\": " << (report.persistent ? "true" : "false") << ",\n";
  os << "  \"programs\": " << report.programs << ",\n";
  os << "  \"mvms\": " << report.mvms << ",\n";
  os << "  \"requantizes\": " << report.requantizes << ",\n";
  os << "  \"reuses\": " << report.reuses << ",\n";
  os << "  \"install_programs\": " << report.installPrograms << ",\n";
  os << "  \"install_energy_pj\": " << report.installEnergyPj << ",\n";
  os << "  \"install_latency_ns\": " << report.installLatencyNs << ",\n";
  os << "  \"steady_state_programs_per_inference\": "
     << report.steadyStateProgramsPerInference << ",\n";
  os << "  \"total_energy_pj\": " << report.totalEnergyPj << ",\n";
  os << "  \"total_latency_ns\": " << report.totalLatencyNs << ",\n";
  os << "  \"standby_leakage_uw_per_tile\": " << report.standbyLeakageUwPerTile
     << ",\n";
  // The per-inference floor a volatile target's amortized install cost
  // settles on and a non-volatile one does not -- see
  // amortizedInstallEnergyPjPerInference's own comment. Zero on a
  // non-volatile target, or on any target declaring zero leakage.
  const double leakageFloorPj = report.persistent
                                     ? 0.0
                                     : static_cast<double>(report.numTiles) *
                                           report.standbyLeakageUwPerTile *
                                           report.steadyStateElapsedNsPerInference *
                                           1e-3;
  os << "  \"standby_leakage_pj_per_inference_floor\": " << leakageFloorPj
     << "\n";
  os << "}\n";
  return os.str();
}

} // namespace cim
