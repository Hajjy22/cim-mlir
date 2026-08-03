//===- test_main.cpp - Minimal test harness --------------------*- C++ -*-===//
//
// Dependency-free so the core tests run anywhere a C++17 compiler exists,
// with no MLIR, no LLVM, and no network to fetch a test framework.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include <cstdio>

namespace cimtest {

std::vector<TestCase> &registry() {
  static std::vector<TestCase> tests;
  return tests;
}

int failureCount = 0;
const char *currentTest = "";

} // namespace cimtest

int main() {
  int failedTests = 0;
  for (const auto &test : cimtest::registry()) {
    cimtest::failureCount = 0;
    cimtest::currentTest = test.name;
    std::printf("[ RUN      ] %s\n", test.name);
    test.fn();
    if (cimtest::failureCount == 0) {
      std::printf("[       OK ] %s\n", test.name);
    } else {
      std::printf("[   FAILED ] %s (%d check(s) failed)\n", test.name,
                  cimtest::failureCount);
      ++failedTests;
    }
  }

  std::printf("\n%zu test(s) run, %d failed\n", cimtest::registry().size(),
              failedTests);
  return failedTests == 0 ? 0 : 1;
}
