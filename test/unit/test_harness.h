//===- test_harness.h - Minimal test harness -------------------*- C++ -*-===//
//
// Dependency-free so the core tests run anywhere a C++17 compiler exists,
// with no MLIR, no LLVM, and no network to fetch a test framework.
//
// Usage:
//   cim-unit-tests [--seed 0xN] [--list] [name-substring ...]
//
// Name filtering exists so a single test can be run under valgrind without
// paying for the whole suite. The seed is reported on every run and is a
// fixed constant by default -- a time-based seed would make failures
// unreproducible, which is worse than no randomization at all.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_TEST_HARNESS_H
#define CIM_TEST_HARNESS_H

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace cimtest {

/// Render any streamable value for a failure message. Keeps CIM_EXPECT_EQ
/// usable on strings and enums, not just arithmetic types.
template <typename T> std::string describe(const T &value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

struct TestCase {
  const char *name;
  void (*fn)();
};

std::vector<TestCase> &registry();
extern int failureCount;
extern const char *currentTest;

/// Seed for this run. Randomized tests must draw from this, never from a
/// clock, so that a failure can be reproduced from the reported value.
extern uint64_t seed;

struct Registrar {
  Registrar(const char *name, void (*fn)()) { registry().push_back({name, fn}); }
};

void reportFailure(const char *file, int line, const std::string &what);

} // namespace cimtest

#define CIM_TEST(name)                                                         \
  static void name();                                                          \
  static ::cimtest::Registrar name##_registrar(#name, name);                   \
  static void name()

#define CIM_EXPECT(cond)                                                       \
  do {                                                                         \
    if (!(cond))                                                               \
      ::cimtest::reportFailure(__FILE__, __LINE__, "expected: " #cond);        \
  } while (0)

#define CIM_FAIL(msg)                                                          \
  ::cimtest::reportFailure(__FILE__, __LINE__, ::cimtest::describe(msg))

/// Binary comparison macros. Each reports both operands on failure, because
/// "expected a == b" without the values is a second debugging session.
#define CIM_EXPECT_BINOP(a, b, op)                                             \
  do {                                                                         \
    auto _lhs = (a);                                                           \
    auto _rhs = (b);                                                           \
    if (!(_lhs op _rhs))                                                       \
      ::cimtest::reportFailure(__FILE__, __LINE__,                             \
                               std::string(#a " " #op " " #b " (got ") +       \
                                   ::cimtest::describe(_lhs) + " vs " +        \
                                   ::cimtest::describe(_rhs) + ")");           \
  } while (0)

#define CIM_EXPECT_EQ(a, b) CIM_EXPECT_BINOP(a, b, ==)
#define CIM_EXPECT_NE(a, b) CIM_EXPECT_BINOP(a, b, !=)
#define CIM_EXPECT_LE(a, b) CIM_EXPECT_BINOP(a, b, <=)
#define CIM_EXPECT_LT(a, b) CIM_EXPECT_BINOP(a, b, <)
#define CIM_EXPECT_GE(a, b) CIM_EXPECT_BINOP(a, b, >=)
#define CIM_EXPECT_GT(a, b) CIM_EXPECT_BINOP(a, b, >)

/// Asserts a diagnostic mentions what it should. Used heavily by the error
/// path tables: a parser that fails for the wrong reason is still a bug.
#define CIM_EXPECT_CONTAINS(haystack, needle)                                  \
  do {                                                                         \
    const std::string _hay = ::cimtest::describe(haystack);                    \
    const std::string _needle = (needle);                                      \
    if (_hay.find(_needle) == std::string::npos)                               \
      ::cimtest::reportFailure(__FILE__, __LINE__,                             \
                               "expected to find \"" + _needle +               \
                                   "\" in \"" + _hay + "\"");                  \
  } while (0)

#endif // CIM_TEST_HARNESS_H
