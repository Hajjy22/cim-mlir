//===- random.h - Deterministic RNG for property tests ---------*- C++ -*-===//
//
// A fixed, self-contained generator rather than std::mt19937, so that a
// seed reproduces the same sequence across platforms and standard-library
// versions. A property test whose failure cannot be reproduced from the
// reported seed is not much better than no test.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_TEST_RANDOM_H
#define CIM_TEST_RANDOM_H

#include <cstdint>
#include <vector>

namespace cimtest {

/// splitmix64: tiny, well-distributed, and fully specified here so results
/// do not depend on the standard library.
class Rng {
public:
  // Parameter is not called `seed`: that would shadow cimtest::seed, the
  // run-wide value this is normally constructed from.
  explicit Rng(uint64_t initialSeed) : state(initialSeed) {}

  uint64_t next() {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  /// Uniform in [lo, hi]. Modulo bias is irrelevant at these range sizes.
  uint32_t range(uint32_t lo, uint32_t hi) {
    if (hi <= lo)
      return lo;
    return lo + static_cast<uint32_t>(next() % (hi - lo + 1));
  }

private:
  uint64_t state;
};

} // namespace cimtest

#endif // CIM_TEST_RANDOM_H
