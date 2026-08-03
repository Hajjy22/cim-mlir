//===- test_main.cpp - Minimal test harness driver -------------*- C++ -*-===//

#include "test_harness.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace cimtest {

std::vector<TestCase> &registry() {
  static std::vector<TestCase> tests;
  return tests;
}

int failureCount = 0;
const char *currentTest = "";

/// Deliberately a fixed constant, not time-based: an unreproducible failure
/// in a randomized test is barely better than no test.
uint64_t seed = 0x5eed1234abcdULL;

void reportFailure(const char *file, int line, const std::string &what) {
  std::printf("  %s:%d: FAIL: %s\n", file, line, what.c_str());
  ++failureCount;
}

} // namespace cimtest

namespace {

void printUsage() {
  std::printf(
      "usage: cim-unit-tests [--seed 0xN] [--list] [name-substring ...]\n\n"
      "  --seed 0xN   seed for randomized tests (default is fixed)\n"
      "  --list       print test names and exit\n"
      "  substrings   run only tests whose name contains one of these\n");
}

bool matchesAnyFilter(const char *name, const std::vector<std::string> &filters) {
  if (filters.empty())
    return true;
  const std::string haystack(name);
  for (const std::string &filter : filters)
    if (haystack.find(filter) != std::string::npos)
      return true;
  return false;
}

} // namespace

int main(int argc, char **argv) {
  std::vector<std::string> filters;
  bool listOnly = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage();
      return 0;
    }
    if (arg == "--list") {
      listOnly = true;
    } else if (arg == "--seed") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "cim-unit-tests: --seed requires a value\n");
        return 2;
      }
      cimtest::seed = std::strtoull(argv[++i], nullptr, 0);
    } else if (arg.rfind("--", 0) == 0) {
      std::fprintf(stderr, "cim-unit-tests: unknown option '%s'\n", arg.c_str());
      return 2;
    } else {
      filters.push_back(arg);
    }
  }

  if (listOnly) {
    for (const auto &test : cimtest::registry())
      std::printf("%s\n", test.name);
    return 0;
  }

  std::printf("[ SEED     ] 0x%llx\n",
              static_cast<unsigned long long>(cimtest::seed));

  int failedTests = 0;
  size_t ran = 0;
  std::vector<const char *> failedNames;

  for (const auto &test : cimtest::registry()) {
    if (!matchesAnyFilter(test.name, filters))
      continue;
    ++ran;
    cimtest::failureCount = 0;
    cimtest::currentTest = test.name;
    std::printf("[ RUN      ] %s\n", test.name);
    test.fn();
    if (cimtest::failureCount == 0) {
      std::printf("[       OK ] %s\n", test.name);
    } else {
      std::printf("[   FAILED ] %s (%d check(s) failed)\n", test.name,
                  cimtest::failureCount);
      failedNames.push_back(test.name);
      ++failedTests;
    }
  }

  std::printf("\n%zu test(s) run, %d failed\n", ran, failedTests);

  // The whole point of the fixed seed: hand back the exact command that
  // reproduces this failure.
  for (const char *name : failedNames)
    std::printf("  re-run with: cim-unit-tests --seed 0x%llx %s\n",
                static_cast<unsigned long long>(cimtest::seed), name);

  return failedTests == 0 ? 0 : 1;
}
