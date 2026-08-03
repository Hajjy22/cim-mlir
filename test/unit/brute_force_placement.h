//===- brute_force_placement.h - Exhaustive optimal placement --*- C++ -*-===//
//
// TEST-ONLY. An exponential-time reference implementation of optimal offline
// replacement, used to check that computePlacement's Belady policy really is
// optimal rather than merely better than LRU.
//
// This is the strongest property available to this project. Belady/MIN is
// *provably* optimal for demand-driven replacement with uniform cost, and
// "optimal placement eliminates reprogramming" is the project's headline
// claim (docs/abstraction.md). Checking Belady against exhaustive search
// turns that claim from an appeal to a textbook result into something a
// machine re-verifies on every commit, against exactly the model
// computePlacement implements: demand-driven, unit cost per program, no
// prefetching, no bypass.
//
// Deliberately in test/ and not lib/: nothing in the shipping compiler may
// ever call this.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_TEST_BRUTE_FORCE_PLACEMENT_H
#define CIM_TEST_BRUTE_FORCE_PLACEMENT_H

#include "cim/Placement/Placement.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace cimtest {

/// Guard against someone quietly making the suite exponential. 2^16 states
/// times a long sequence is already seconds; beyond that it is minutes.
inline constexpr unsigned kMaxDistinctWeights = 16;

/// Minimum number of programs (cache misses) achievable by ANY offline
/// replacement policy for this use sequence and tile count.
///
/// Dynamic programming over the set of resident weights. Tile *identity* is
/// irrelevant -- every program costs the same and any weight can go in any
/// tile -- so the state is just "which weights are resident", a bitmask.
/// That collapses what would be a T-dimensional assignment problem into
/// 2^D states.
///
///   dp[mask] = fewest programs to arrive at residency `mask` after the
///              prefix processed so far
///
/// Per step, with w = seq[i]:
///   - w already in mask         -> free, mask unchanged
///   - room left (popcount < T)  -> +1 program, add w
///   - otherwise                 -> +1 program, evict each candidate v in
///                                  turn (this is where every eviction
///                                  choice, not just Belady's, gets tried)
inline uint64_t optimalPrograms(const std::vector<cim::WeightId> &seq,
                                 uint32_t numTiles) {
  if (seq.empty() || numTiles == 0)
    return 0;

  // Renumber weights densely so a bitmask stays small.
  std::unordered_map<cim::WeightId, unsigned> denseId;
  std::vector<unsigned> dense;
  dense.reserve(seq.size());
  for (cim::WeightId w : seq) {
    auto [it, inserted] = denseId.try_emplace(w, static_cast<unsigned>(denseId.size()));
    dense.push_back(it->second);
  }
  const unsigned distinct = static_cast<unsigned>(denseId.size());
  assert(distinct <= kMaxDistinctWeights &&
         "optimalPrograms is exponential in the number of distinct weights");

  // More tiles than weights: everything fits, so the answer is trivially
  // one program per distinct weight. Skipping the DP here keeps the state
  // space from being needlessly large for the easy case.
  if (numTiles >= distinct)
    return distinct;

  constexpr uint64_t kInf = std::numeric_limits<uint64_t>::max();
  const size_t numStates = size_t{1} << distinct;

  std::vector<uint64_t> dp(numStates, kInf);
  std::vector<uint64_t> next(numStates, kInf);
  dp[0] = 0;

  for (unsigned w : dense) {
    std::fill(next.begin(), next.end(), kInf);
    const uint32_t bit = uint32_t{1} << w;

    for (size_t mask = 0; mask < numStates; ++mask) {
      const uint64_t cost = dp[mask];
      if (cost == kInf)
        continue;

      if (mask & bit) {
        // Resident: reuse, no program emitted.
        next[mask] = std::min(next[mask], cost);
        continue;
      }

      const unsigned resident =
          static_cast<unsigned>(__builtin_popcountll(static_cast<uint64_t>(mask)));
      if (resident < numTiles) {
        const size_t added = mask | bit;
        next[added] = std::min(next[added], cost + 1);
        continue;
      }

      // Full: try evicting every resident weight in turn.
      for (unsigned v = 0; v < distinct; ++v) {
        const uint32_t vbit = uint32_t{1} << v;
        if (!(mask & vbit))
          continue;
        const size_t evicted = (mask & ~static_cast<size_t>(vbit)) | bit;
        next[evicted] = std::min(next[evicted], cost + 1);
      }
    }
    dp.swap(next);
  }

  uint64_t best = kInf;
  for (uint64_t cost : dp)
    best = std::min(best, cost);
  return best;
}

} // namespace cimtest

#endif // CIM_TEST_BRUTE_FORCE_PLACEMENT_H
