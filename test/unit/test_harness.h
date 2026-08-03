//===- test_harness.h - Minimal test harness -------------------*- C++ -*-===//
#ifndef CIM_TEST_HARNESS_H
#define CIM_TEST_HARNESS_H

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

struct Registrar {
  Registrar(const char *name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline void reportFailure(const char *file, int line, const std::string &what) {
  std::printf("  %s:%d: FAIL: %s\n", file, line, what.c_str());
  ++failureCount;
}

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

#define CIM_EXPECT_EQ(a, b)                                                    \
  do {                                                                         \
    auto _lhs = (a);                                                           \
    auto _rhs = (b);                                                           \
    if (!(_lhs == _rhs))                                                       \
      ::cimtest::reportFailure(__FILE__, __LINE__,                             \
                               std::string(#a " == " #b " (got ") +            \
                                   ::cimtest::describe(_lhs) + " vs " +          \
                                   ::cimtest::describe(_rhs) + ")");                \
  } while (0)

#define CIM_EXPECT_LE(a, b)                                                    \
  do {                                                                         \
    auto _lhs = (a);                                                           \
    auto _rhs = (b);                                                           \
    if (!(_lhs <= _rhs))                                                       \
      ::cimtest::reportFailure(__FILE__, __LINE__,                             \
                               std::string(#a " <= " #b " (got ") +            \
                                   ::cimtest::describe(_lhs) + " vs " +          \
                                   ::cimtest::describe(_rhs) + ")");                \
  } while (0)

#endif // CIM_TEST_HARNESS_H
