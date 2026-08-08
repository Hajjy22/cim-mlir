//===- brute_force_steady_state.h - Exhaustive pin choice -------*- C++ -*-===//
//
// TEST-ONLY. An exponential-time reference for the specific question
// cim::computeSteadyStatePlacement answers: given one loop iteration's use
// sequence and a fixed set of `numTiles - 1` pinned weights, streaming
// everything else through the one remaining tile costs a fully determined
// number of programs (simulate the single streaming tile directly -- no
// choice of victim exists with one tile, so there is nothing to optimize
// once the pinned set is fixed). What is NOT determined in advance is
// *which* weights to pin, and this enumerates every possible choice to find
// the true minimum, the same way brute_force_placement.h enumerates every
// eviction choice to find computePlacement's.
//
// Deliberately in test/ and not lib/: nothing in the shipping compiler may
// ever call this.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_TEST_BRUTE_FORCE_STEADY_STATE_H
#define CIM_TEST_BRUTE_FORCE_STEADY_STATE_H

#include "cim/Placement/Placement.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace cimtest {

/// Guard against someone quietly making the suite exponential:
/// C(distinctWeights, pinnedCount) grows fast, and this is only ever run
/// against small, hand- or randomly-generated instances.
inline constexpr unsigned kMaxDistinctForBruteForceSteadyState = 12;

/// Programs-per-iteration cost of streaming `seq` through one tile with
/// `pinned` held permanently resident elsewhere -- exactly the simulation
/// cim::computeSteadyStatePlacement itself runs (a single streaming tile
/// has no choice of victim: consecutive repeats of the same non-pinned
/// weight reuse it, anything else reprograms it), duplicated here so the
/// brute-force search below can score an arbitrary candidate pinned set
/// without depending on the function under test.
inline uint64_t
streamingCost(const std::vector<cim::WeightId> &seq,
             const std::unordered_set<cim::WeightId> &pinned) {
  uint64_t programs = 0;
  cim::WeightId resident = cim::kNoWeight;
  for (cim::WeightId w : seq) {
    if (pinned.count(w))
      continue;
    if (w != resident) {
      ++programs;
      resident = w;
    }
  }
  return programs;
}

/// Fewest possible programs per iteration achievable by ANY fixed choice
/// of `numTiles - 1` permanently-pinned weights, streaming everything else
/// through the one remaining tile. This is the exact question
/// cim::computeSteadyStatePlacement's occurrence-count heuristic answers
/// approximately when a weight repeats within `seq` (see that function's
/// own header comment on when it is exact vs. a heuristic).
inline uint64_t optimalSteadyStateProgramsPerIteration(
    const std::vector<cim::WeightId> &seq, uint32_t numTiles) {
  std::vector<cim::WeightId> distinct;
  {
    std::unordered_set<cim::WeightId> seen;
    for (cim::WeightId w : seq)
      if (seen.insert(w).second)
        distinct.push_back(w);
  }
  assert(distinct.size() <= kMaxDistinctForBruteForceSteadyState &&
        "optimalSteadyStateProgramsPerIteration is exponential in the "
        "number of distinct weights");

  if (numTiles == 0)
    return 0; // unsatisfiable; caller's problem, not this function's.
  if (distinct.size() <= numTiles)
    return 0; // fits entirely: pin everything, nothing streams.

  const uint32_t pinnedCount = numTiles - 1;
  const unsigned n = static_cast<unsigned>(distinct.size());

  uint64_t best = UINT64_MAX;
  // Enumerate every pinnedCount-sized subset of distinct as a bitmask.
  // n <= kMaxDistinctForBruteForceSteadyState keeps 2^n small.
  const unsigned total = 1u << n;
  for (unsigned mask = 0; mask < total; ++mask) {
    if (static_cast<unsigned>(__builtin_popcount(mask)) != pinnedCount)
      continue;
    std::unordered_set<cim::WeightId> pinned;
    for (unsigned i = 0; i < n; ++i)
      if (mask & (1u << i))
        pinned.insert(distinct[i]);
    best = std::min(best, streamingCost(seq, pinned));
  }
  return best;
}

} // namespace cimtest

#endif // CIM_TEST_BRUTE_FORCE_STEADY_STATE_H
