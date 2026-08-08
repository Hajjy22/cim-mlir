//===- cim-bench.cpp - Benchmark harness (spec Sec. 10) --------*- C++ -*-===//
//
//   cim-bench run --target erbium-8t --out results.json
//
// Harness requirements from spec Sec. 10, and where each is met:
//   - one command runs everything            -> the `run` subcommand below
//   - results carry target hash, commit, date -> emitProvenance()
//   - correctness check beside every number  -> validatePlacement() per row
//   - plot scripts live in the repo          -> bench/plots/
//
// Measures weight-residency cost only: how many cim.program ops each
// eviction policy needs for a given workload, and what the target's cost
// table says that adds up to. Numerical correctness of the arithmetic is
// the functional simulator's job (runtime/src/simulator), not this tool's.
//
// No LLVM/MLIR dependency, so the benchmark runs without a toolchain build.
//
//===----------------------------------------------------------------------===//

#include "cim/Placement/CostReport.h"
#include "cim/Placement/Placement.h"
#include "cim/Placement/Workloads.h"
#include "cim/Target/TargetSpec.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace cim;

namespace {

struct Options {
  std::string command = "run";
  std::string target;
  std::string targetFile;
  std::string outPath = "results.json";
  uint32_t inferences = 1000;
  std::string workload = "mm-fit";
  std::string policy = "belady";
};

void printUsage() {
  std::cout
      << "cim-bench: cim-mlir benchmark harness\n\n"
         "usage: cim-bench run --target <name> [--out <path>] [--inferences <n>]\n"
         "       cim-bench dump-target --target <name>\n"
         "       cim-bench amortize --target <name> [--workload <name>] "
         "[--policy <p>] [--out <path>]\n\n"
         "  dump-target        print the parsed target as canonical key=value\n"
         "                     lines, for differential-testing the reader\n"
         "  amortize           sweep amortizedInstallEnergyPjPerInference over a\n"
         "                     fixed range of inference counts for one workload,\n"
         "                     to plot how install cost amortizes -- or, on a\n"
         "                     volatile target, does not (docs/roadmap.md's M3\n"
         "                     section; bench/plots/plot_amortization.py)\n"
         "  --target <name>    target name; resolved to targets/<name>.yaml\n"
         "  --target-file <p>  explicit path to a target file (overrides --target)\n"
         "  --out <path>       results JSON destination (default results.json)\n"
         "  --inferences <n>   inferences to model per workload (default 1000)\n"
         "  --workload <name>  amortize only: one of mm-fit, mm-spill-2x,\n"
         "                     mm-spill-8x, mlp-3layer, bert-ffn (default mm-fit)\n"
         "  --policy <p>       amortize only: belady, lru, or fifo (default belady)\n";
}

bool parseArgs(int argc, char **argv, Options &opts) {
  if (argc < 2) {
    printUsage();
    return false;
  }
  opts.command = argv[1];
  if (opts.command == "-h" || opts.command == "--help" || opts.command == "help") {
    printUsage();
    return false;
  }
  if (opts.command != "run" && opts.command != "dump-target" &&
      opts.command != "amortize") {
    std::cerr << "cim-bench: unknown command '" << opts.command
              << "' (expected 'run', 'dump-target', or 'amortize')\n";
    return false;
  }

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char *what) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "cim-bench: " << what << " requires a value\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--target") {
      const char *v = next("--target");
      if (!v) return false;
      opts.target = v;
    } else if (arg == "--target-file") {
      const char *v = next("--target-file");
      if (!v) return false;
      opts.targetFile = v;
    } else if (arg == "--out") {
      const char *v = next("--out");
      if (!v) return false;
      opts.outPath = v;
    } else if (arg == "--inferences") {
      const char *v = next("--inferences");
      if (!v) return false;
      opts.inferences = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
      if (opts.inferences == 0) {
        std::cerr << "cim-bench: --inferences must be at least 1\n";
        return false;
      }
    } else if (arg == "--workload") {
      const char *v = next("--workload");
      if (!v) return false;
      opts.workload = v;
    } else if (arg == "--policy") {
      const char *v = next("--policy");
      if (!v) return false;
      opts.policy = v;
    } else {
      std::cerr << "cim-bench: unrecognized option '" << arg << "'\n";
      return false;
    }
  }

  if (opts.target.empty() && opts.targetFile.empty()) {
    std::cerr << "cim-bench: --target (or --target-file) is required\n";
    return false;
  }
  if (opts.targetFile.empty())
    opts.targetFile = "targets/" + opts.target + ".yaml";
  if (opts.target.empty())
    opts.target = opts.targetFile;

  return true;
}

/// FNV-1a over the target file's bytes. Identifies exactly which cost
/// numbers produced a result, so a plot can never be silently attributed to
/// a target file that has since been edited (spec Sec. 10).
std::string hashFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return "unavailable";
  uint64_t hash = 1469598103934665603ULL;
  char c;
  while (in.get(c)) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ULL;
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
  return buf;
}

std::string gitCommit() {
  FILE *pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
  if (!pipe)
    return "unknown";
  char buf[64] = {0};
  const char *read = std::fgets(buf, sizeof(buf), pipe);
  pclose(pipe);
  if (!read)
    return "unknown";
  std::string commit(buf);
  while (!commit.empty() && (commit.back() == '\n' || commit.back() == '\r'))
    commit.pop_back();
  return commit.empty() ? "unknown" : commit;
}

std::string utcTimestamp() {
  std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

const char *targetClassName(TargetClass c) {
  switch (c) {
  case TargetClass::NearMemory: return "near_memory";
  case TargetClass::DigitalCIM: return "digital_cim";
  case TargetClass::AnalogCIM: return "analog_cim";
  case TargetClass::DPU: return "dpu";
  }
  return "unknown";
}

const char *provenanceName(Provenance p) {
  switch (p) {
  case Provenance::Measured: return "measured";
  case Provenance::Simulated: return "simulated";
  case Provenance::Estimated: return "estimated";
  }
  return "unknown";
}

struct Row {
  std::string workload;
  std::string probes;
  std::string policy;
  bool correct = false;
  uint64_t programs = 0;
  uint64_t reuses = 0;
  uint64_t installPrograms = 0;
  uint64_t steadyPrograms = 0;
  double totalEnergyPj = 0.0;
  double amortizedInstallPjPerInference = 0.0;
};

} // namespace

int main(int argc, char **argv) {
  Options opts;
  if (!parseArgs(argc, argv, opts))
    return 1;

  TargetSpec spec;
  std::string error;
  if (!parseTargetSpecFromFile(opts.targetFile, spec, &error)) {
    std::cerr << "cim-bench: " << error << "\n";
    return 1;
  }

  if (opts.command == "dump-target") {
    // Canonical, sorted, one field per line. Deliberately not JSON: the
    // point is a format a differential test can compare field by field
    // without a parser of its own being a source of disagreement.
    std::printf("name=%s\n", spec.name.c_str());
    std::printf("class=%s\n", targetClassName(spec.targetClass));
    std::printf("provenance=%s\n", provenanceName(spec.provenance));
    std::printf("tiles.count=%u\n", spec.tiles.count);
    std::printf("tiles.rows=%u\n", spec.tiles.rows);
    std::printf("tiles.cols=%u\n", spec.tiles.cols);
    std::printf("tiles.weight_dtype=%s\n", spec.tiles.weightDtype.c_str());
    std::printf("tiles.activation_dtype=%s\n",
                spec.tiles.activationDtype.c_str());
    std::printf("tiles.accumulator_dtype=%s\n",
                spec.tiles.accumulatorDtype.c_str());
    std::printf("tiles.persistent=%s\n", spec.tiles.persistent ? "true" : "false");
    std::printf("tiles.persistence=%s\n", spec.tiles.persistence.c_str());
    std::printf("costs.program.latency_ns=%.10g\n", spec.costs.program.latencyNs);
    std::printf("costs.program.energy_pj=%.10g\n", spec.costs.program.energyPj);
    std::printf("costs.mvm.latency_ns=%.10g\n", spec.costs.mvm.latencyNs);
    std::printf("costs.mvm.energy_pj=%.10g\n", spec.costs.mvm.energyPj);
    std::printf("costs.transfer.bandwidth_gbps=%.10g\n",
                spec.costs.transfer.bandwidthGbps);
    std::printf("costs.transfer.energy_pj_per_byte=%.10g\n",
                spec.costs.transfer.energyPjPerByte);
    std::printf("costs.standby_leakage_uw_per_tile=%.10g\n",
                spec.costs.standbyLeakageUwPerTile);
    std::printf("precision.output_effective_bits=%u\n",
                spec.precision.outputEffectiveBits);
    std::printf("capabilities.double_buffer_program=%s\n",
                spec.capabilities.doubleBufferProgram ? "true" : "false");
    std::printf("capabilities.partial_sum_in_place=%s\n",
                spec.capabilities.partialSumInPlace ? "true" : "false");
    std::printf("capabilities.autonomous_control=%s\n",
                spec.capabilities.autonomousControl ? "true" : "false");
    return 0;
  }

  if (opts.command == "amortize") {
    if (spec.provenance != Provenance::Measured) {
      std::cerr << "cim-bench: WARNING: target '" << spec.name << "' declares "
                << provenanceName(spec.provenance)
                << " costs, not measured. Results below inherit that "
                   "uncertainty and must be labelled as such in any plot.\n\n";
    }

    EvictionPolicy policy;
    if (opts.policy == "belady") {
      policy = EvictionPolicy::Belady;
    } else if (opts.policy == "lru") {
      policy = EvictionPolicy::LRU;
    } else if (opts.policy == "fifo") {
      policy = EvictionPolicy::FIFO;
    } else {
      std::cerr << "cim-bench: unknown policy '" << opts.policy
                << "' (expected belady, lru, or fifo)\n";
      return 1;
    }

    const std::vector<Workload> workloads = makeV01Workloads(
        spec.tiles.count, spec.tiles.rows, spec.tiles.cols, opts.inferences);
    const Workload *wl = nullptr;
    for (const Workload &w : workloads)
      if (w.name == opts.workload) {
        wl = &w;
        break;
      }
    if (!wl) {
      std::cerr << "cim-bench: unknown workload '" << opts.workload
                << "' (expected one of: mm-fit, mm-spill-2x, mm-spill-8x, "
                   "mlp-3layer, bert-ffn)\n";
      return 1;
    }

    PlacementResult result = computePlacement(wl->problem, policy);
    std::string validationError;
    if (!validatePlacement(wl->problem, result, &validationError)) {
      std::cerr << "cim-bench: INVALID SCHEDULE for " << wl->name << "/"
                << opts.policy << ": " << validationError << "\n";
      return 1;
    }
    const CostReport report = computeCostReport(spec, result, wl->stepsPerInference);

    // A fixed, wide log-spaced sweep, not one derived from --inferences:
    // amortizedInstallEnergyPjPerInference is a closed-form function of the
    // CostReport above (itself built from one representative run) and an
    // arbitrary inference count, so one report answers the whole sweep --
    // see the function's own header comment. Re-running placement per sweep
    // point would not change a single number here and would not scale to
    // the high end of this range regardless (mm-fit's use sequence alone
    // would need 8 * 10^8 entries to actually simulate 10^8 inferences).
    static const uint64_t kSweep[] = {1,      10,      100,       1000,
                                      10000,  100000,  1000000,   10000000,
                                      100000000, 1000000000};

    std::ofstream out(opts.outPath);
    if (!out) {
      std::cerr << "cim-bench: failed to open '" << opts.outPath
                << "' for writing\n";
      return 1;
    }

    const std::string targetHash = hashFile(opts.targetFile);
    out << "{\n";
    out << "  \"target\": \"" << spec.name << "\",\n";
    out << "  \"target_file\": \"" << opts.targetFile << "\",\n";
    out << "  \"target_file_hash\": \"" << targetHash << "\",\n";
    out << "  \"provenance\": \"" << provenanceName(spec.provenance) << "\",\n";
    out << "  \"git_commit\": \"" << gitCommit() << "\",\n";
    out << "  \"date\": \"" << utcTimestamp() << "\",\n";
    out << "  \"workload\": \"" << wl->name << "\",\n";
    out << "  \"policy\": \"" << toString(policy) << "\",\n";
    out << "  \"persistent\": " << (spec.tiles.persistent ? "true" : "false") << ",\n";
    out << "  \"standby_leakage_uw_per_tile\": "
        << spec.costs.standbyLeakageUwPerTile << ",\n";
    out << "  \"install_energy_pj\": " << report.installEnergyPj << ",\n";
    out << "  \"points\": [\n";
    const size_t numSweep = sizeof(kSweep) / sizeof(kSweep[0]);
    for (size_t i = 0; i < numSweep; ++i) {
      const uint64_t n = kSweep[i];
      const double perInf = amortizedInstallEnergyPjPerInference(report, n);
      out << "    {\"inferences\": " << n
          << ", \"amortized_install_pj_per_inference\": " << perInf << "}"
          << (i + 1 < numSweep ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    std::printf("wrote %s (target %s, workload %s, policy %s, hash %s)\n",
                opts.outPath.c_str(), spec.name.c_str(), wl->name.c_str(),
                opts.policy.c_str(), targetHash.c_str());
    return 0;
  }

  if (spec.provenance != Provenance::Measured) {
    // Spec Sec. 7: label estimates loudly. A reader must never mistake these
    // numbers for measurements.
    std::cerr << "cim-bench: WARNING: target '" << spec.name << "' declares "
              << provenanceName(spec.provenance)
              << " costs, not measured. Results below inherit that "
                 "uncertainty and must be labelled as such in any plot.\n\n";
  }

  const std::vector<Workload> workloads =
      makeV01Workloads(spec.tiles.count, spec.tiles.rows, spec.tiles.cols,
                       opts.inferences);

  const EvictionPolicy policies[] = {EvictionPolicy::Belady, EvictionPolicy::LRU,
                                      EvictionPolicy::FIFO};

  std::vector<Row> rows;
  bool allCorrect = true;

  std::printf("%-14s %-8s %6s %10s %10s %12s\n", "workload", "policy", "ok",
              "programs", "install", "steady/inf");
  std::printf("%s\n", std::string(66, '-').c_str());

  for (const Workload &wl : workloads) {
    for (EvictionPolicy policy : policies) {
      PlacementResult result = computePlacement(wl.problem, policy);

      // Correctness check beside the performance number, always.
      std::string validationError;
      const bool ok = validatePlacement(wl.problem, result, &validationError);
      if (!ok) {
        allCorrect = false;
        std::fprintf(stderr, "cim-bench: INVALID SCHEDULE for %s/%s: %s\n",
                     wl.name.c_str(), toString(policy), validationError.c_str());
      }

      CostReport report = computeCostReport(spec, result, wl.stepsPerInference);

      Row row;
      row.workload = wl.name;
      row.probes = wl.probes;
      row.policy = toString(policy);
      row.correct = ok;
      row.programs = result.programs;
      row.reuses = result.reuses;
      row.installPrograms = report.installPrograms;
      row.steadyPrograms = report.steadyStateProgramsPerInference;
      row.totalEnergyPj = report.totalEnergyPj;
      row.amortizedInstallPjPerInference =
          amortizedInstallEnergyPjPerInference(report, opts.inferences);
      rows.push_back(row);

      std::printf("%-14s %-8s %6s %10llu %10llu %12llu\n", wl.name.c_str(),
                  row.policy.c_str(), ok ? "yes" : "NO",
                  static_cast<unsigned long long>(row.programs),
                  static_cast<unsigned long long>(row.installPrograms),
                  static_cast<unsigned long long>(row.steadyPrograms));
    }
  }

  const std::string targetHash = hashFile(opts.targetFile);
  const std::string commit = gitCommit();
  const std::string timestamp = utcTimestamp();

  std::ofstream out(opts.outPath);
  if (!out) {
    std::cerr << "cim-bench: failed to open '" << opts.outPath << "' for writing\n";
    return 1;
  }

  out << "{\n";
  out << "  \"target\": \"" << spec.name << "\",\n";
  out << "  \"target_file\": \"" << opts.targetFile << "\",\n";
  out << "  \"target_file_hash\": \"" << targetHash << "\",\n";
  out << "  \"provenance\": \"" << provenanceName(spec.provenance) << "\",\n";
  out << "  \"git_commit\": \"" << commit << "\",\n";
  out << "  \"date\": \"" << timestamp << "\",\n";
  out << "  \"inferences_modeled\": " << opts.inferences << ",\n";
  out << "  \"all_schedules_valid\": " << (allCorrect ? "true" : "false") << ",\n";
  out << "  \"results\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const Row &r = rows[i];
    out << "    {\n";
    out << "      \"workload\": \"" << r.workload << "\",\n";
    out << "      \"probes\": \"" << r.probes << "\",\n";
    out << "      \"policy\": \"" << r.policy << "\",\n";
    out << "      \"schedule_valid\": " << (r.correct ? "true" : "false") << ",\n";
    out << "      \"programs\": " << r.programs << ",\n";
    out << "      \"reuses\": " << r.reuses << ",\n";
    out << "      \"install_programs\": " << r.installPrograms << ",\n";
    out << "      \"steady_state_programs_per_inference\": " << r.steadyPrograms << ",\n";
    out << "      \"total_energy_pj\": " << r.totalEnergyPj << ",\n";
    out << "      \"amortized_install_pj_per_inference\": "
        << r.amortizedInstallPjPerInference << "\n";
    out << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
  }
  out << "  ]\n";
  out << "}\n";

  std::printf("\nwrote %s (target %s, hash %s, commit %s)\n", opts.outPath.c_str(),
              spec.name.c_str(), targetHash.c_str(), commit.c_str());

  if (!allCorrect) {
    std::fprintf(stderr,
                 "cim-bench: at least one schedule failed validation; "
                 "performance numbers above are not trustworthy\n");
    return 1;
  }
  return 0;
}
