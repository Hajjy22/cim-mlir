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

  return report;
}

double amortizedInstallEnergyPjPerInference(const CostReport &report,
                                             uint64_t inferences) {
  if (inferences == 0)
    return 0.0;
  return report.installEnergyPj / static_cast<double>(inferences);
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
  os << "  reuses:    " << report.reuses << " (steps needing no reprogram)\n";

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
  os << "  \"reuses\": " << report.reuses << ",\n";
  os << "  \"install_programs\": " << report.installPrograms << ",\n";
  os << "  \"install_energy_pj\": " << report.installEnergyPj << ",\n";
  os << "  \"install_latency_ns\": " << report.installLatencyNs << ",\n";
  os << "  \"steady_state_programs_per_inference\": "
     << report.steadyStateProgramsPerInference << ",\n";
  os << "  \"total_energy_pj\": " << report.totalEnergyPj << ",\n";
  os << "  \"total_latency_ns\": " << report.totalLatencyNs << "\n";
  os << "}\n";
  return os.str();
}

} // namespace cim
