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
};

void printUsage() {
  std::cout
      << "cim-bench: cim-mlir benchmark harness\n\n"
         "usage: cim-bench run --target <name> [--out <path>] [--inferences <n>]\n\n"
         "  --target <name>    target name; resolved to targets/<name>.yaml\n"
         "  --target-file <p>  explicit path to a target file (overrides --target)\n"
         "  --out <path>       results JSON destination (default results.json)\n"
         "  --inferences <n>   inferences to model per workload (default 1000)\n";
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
  if (opts.command != "run") {
    std::cerr << "cim-bench: unknown command '" << opts.command
              << "' (only 'run' is implemented)\n";
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
