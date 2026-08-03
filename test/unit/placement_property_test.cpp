//===- placement_property_test.cpp - Placement invariants ------*- C++ -*-===//
//
// Property-based and exhaustive checks on computePlacement. The headline
// one is belady_is_optimal_*: it verifies against exhaustive search that the
// Belady policy really achieves the minimum possible number of cim.program
// ops, which is the claim docs/abstraction.md rests the whole project on.
//
// Every randomized failure reports its seed and a shrunk counterexample, so
// it can be reproduced and promoted to a permanent regression test.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "brute_force_placement.h"
#include "random.h"
#include "shrink.h"

#include "cim/Placement/Placement.h"

#include <string>
#include <vector>

using namespace cim;
using namespace cimtest;

namespace {

/// Enumerate every sequence of `length` symbols drawn from `alphabet`.
/// Complete coverage of a small instance class beats sampling a large one:
/// if Belady is going to disagree with optimal, it will do so on a tiny
/// adversarial pattern, and this finds all of them.
void forEachSequence(unsigned alphabet, unsigned length,
                     const std::function<void(const std::vector<WeightId> &)> &fn) {
  std::vector<WeightId> seq(length, 0);
  const uint64_t total = [&] {
    uint64_t t = 1;
    for (unsigned i = 0; i < length; ++i)
      t *= alphabet;
    return t;
  }();

  for (uint64_t code = 0; code < total; ++code) {
    uint64_t rest = code;
    for (unsigned i = 0; i < length; ++i) {
      seq[i] = static_cast<WeightId>(rest % alphabet);
      rest /= alphabet;
    }
    fn(seq);
  }
}

PlacementProblem makeProblem(std::vector<WeightId> seq, uint32_t numTiles) {
  PlacementProblem problem;
  problem.useSequence = std::move(seq);
  problem.numTiles = numTiles;
  problem.name = "property";
  return problem;
}

/// Random instance with a small enough alphabet for the exhaustive oracle.
PlacementProblem randomProblem(Rng &rng, unsigned maxDistinct, unsigned maxLen) {
  const uint32_t distinct = rng.range(1, maxDistinct);
  const uint32_t length = rng.range(1, maxLen);
  std::vector<WeightId> seq;
  seq.reserve(length);
  for (uint32_t i = 0; i < length; ++i)
    seq.push_back(static_cast<WeightId>(rng.range(0, distinct - 1)));
  return makeProblem(std::move(seq), rng.range(1, distinct));
}

/// Run `prop` over random instances; on failure, shrink and report with the
/// seed so the case can be reproduced exactly.
void runProperty(const char *name, unsigned iterations, unsigned maxDistinct,
                 unsigned maxLen, const PlacementProperty &prop) {
  Rng rng(cimtest::seed);
  for (unsigned i = 0; i < iterations; ++i) {
    PlacementProblem problem = randomProblem(rng, maxDistinct, maxLen);
    std::string why;
    if (prop(problem, &why))
      continue;

    const PlacementProblem minimal = shrinkCounterexample(problem, prop);
    std::string minimalWhy;
    prop(minimal, &minimalWhy);
    CIM_FAIL(std::string(name) + " failed: " + minimalWhy +
             "\n    minimal counterexample: " + formatProblem(minimal) +
             "\n    (original: " + formatProblem(problem) + ")");
    return; // one clear report beats hundreds of correlated ones
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// The headline property: Belady achieves the true optimum
//===----------------------------------------------------------------------===//

CIM_TEST(belady_is_optimal_exhaustively_on_small_instances) {
  // Every sequence over a 3-symbol alphabet up to length 7, for 1 and 2
  // tiles. Not sampled -- complete.
  unsigned checked = 0;
  for (unsigned length = 1; length <= 7; ++length) {
    forEachSequence(3, length, [&](const std::vector<WeightId> &seq) {
      for (uint32_t tiles : {1u, 2u}) {
        const PlacementProblem problem = makeProblem(seq, tiles);
        const PlacementResult result =
            computePlacement(problem, EvictionPolicy::Belady);
        const uint64_t best = optimalPrograms(problem.useSequence, tiles);
        if (result.programs != best) {
          CIM_FAIL("Belady emitted " + std::to_string(result.programs) +
                   " programs but the optimum is " + std::to_string(best) +
                   " for " + formatProblem(problem));
          return;
        }
        ++checked;
      }
    });
  }
  // Guard against the loop silently doing nothing.
  CIM_EXPECT_GT(checked, 4000u);
}

CIM_TEST(belady_is_optimal_on_random_instances) {
  runProperty("belady == optimal", /*iterations=*/500, /*maxDistinct=*/10,
              /*maxLen=*/20,
              [](const PlacementProblem &problem, std::string *why) {
                const PlacementResult result =
                    computePlacement(problem, EvictionPolicy::Belady);
                const uint64_t best =
                    optimalPrograms(problem.useSequence, problem.numTiles);
                if (result.programs == best)
                  return true;
                *why = "Belady emitted " + std::to_string(result.programs) +
                       " programs, optimum is " + std::to_string(best);
                return false;
              });
}

//===----------------------------------------------------------------------===//
// Schedule validity and accounting invariants
//===----------------------------------------------------------------------===//

CIM_TEST(every_policy_produces_a_valid_schedule) {
  for (EvictionPolicy policy :
       {EvictionPolicy::Belady, EvictionPolicy::LRU, EvictionPolicy::FIFO}) {
    runProperty("schedule validity", 300, 12, 30,
                [policy](const PlacementProblem &problem, std::string *why) {
                  const PlacementResult result = computePlacement(problem, policy);
                  std::string error;
                  if (validatePlacement(problem, result, &error))
                    return true;
                  *why = std::string(toString(policy)) + ": " + error;
                  return false;
                });
  }
}

CIM_TEST(programs_plus_reuses_accounts_for_every_step) {
  for (EvictionPolicy policy :
       {EvictionPolicy::Belady, EvictionPolicy::LRU, EvictionPolicy::FIFO}) {
    runProperty("accounting", 300, 12, 30,
                [policy](const PlacementProblem &problem, std::string *why) {
                  const PlacementResult r = computePlacement(problem, policy);
                  const uint64_t steps = problem.useSequence.size();
                  if (r.programs + r.reuses != steps) {
                    *why = "programs + reuses != steps";
                    return false;
                  }
                  if (r.programs < r.distinctWeights) {
                    *why = "fewer programs than distinct weights: a weight "
                           "was used without ever being programmed";
                    return false;
                  }
                  if (r.evictions > r.programs) {
                    *why = "more evictions than programs";
                    return false;
                  }
                  return true;
                });
  }
}

CIM_TEST(a_model_that_fits_never_evicts) {
  runProperty("fits => no eviction", 300, 10, 30,
              [](const PlacementProblem &problem, std::string *why) {
                PlacementProblem roomy = problem;
                // Give it strictly more tiles than distinct weights.
                const PlacementResult probe =
                    computePlacement(problem, EvictionPolicy::Belady);
                roomy.numTiles = static_cast<uint32_t>(probe.distinctWeights);
                const PlacementResult r =
                    computePlacement(roomy, EvictionPolicy::Belady);
                if (r.evictions != 0) {
                  *why = "evicted despite the model fitting in tiles";
                  return false;
                }
                if (r.programs != r.distinctWeights) {
                  *why = "programmed " + std::to_string(r.programs) +
                         " times for " + std::to_string(r.distinctWeights) +
                         " distinct weights that all fit";
                  return false;
                }
                return true;
              });
}

//===----------------------------------------------------------------------===//
// Policy comparison
//===----------------------------------------------------------------------===//

CIM_TEST(belady_never_loses_to_lru_or_fifo) {
  runProperty("belady <= baselines", 500, 12, 30,
              [](const PlacementProblem &problem, std::string *why) {
                const uint64_t belady =
                    computePlacement(problem, EvictionPolicy::Belady).programs;
                const uint64_t lru =
                    computePlacement(problem, EvictionPolicy::LRU).programs;
                const uint64_t fifo =
                    computePlacement(problem, EvictionPolicy::FIFO).programs;
                if (belady > lru) {
                  *why = "belady " + std::to_string(belady) + " > lru " +
                         std::to_string(lru);
                  return false;
                }
                if (belady > fifo) {
                  *why = "belady " + std::to_string(belady) + " > fifo " +
                         std::to_string(fifo);
                  return false;
                }
                return true;
              });
}

CIM_TEST(more_tiles_never_hurt_belady_or_lru) {
  // Both Belady and LRU are stack algorithms, so adding a tile can never
  // increase the number of misses.
  //
  // This is deliberately NOT asserted for FIFO: FIFO is not a stack
  // algorithm and exhibits Belady's anomaly, where more tiles can mean MORE
  // misses. Asserting monotonicity there would be a plausible-looking test
  // that fails on correct code.
  for (EvictionPolicy policy : {EvictionPolicy::Belady, EvictionPolicy::LRU}) {
    runProperty("tile monotonicity", 200, 10, 25,
                [policy](const PlacementProblem &problem, std::string *why) {
                  PlacementProblem more = problem;
                  more.numTiles = problem.numTiles + 1;
                  const uint64_t fewer =
                      computePlacement(problem, policy).programs;
                  const uint64_t extra = computePlacement(more, policy).programs;
                  if (extra <= fewer)
                    return true;
                  *why = std::string(toString(policy)) + ": " +
                         std::to_string(problem.numTiles) + " tiles -> " +
                         std::to_string(fewer) + " programs, but " +
                         std::to_string(more.numTiles) + " tiles -> " +
                         std::to_string(extra);
                  return false;
                });
  }
}

//===----------------------------------------------------------------------===//
// The oracle itself
//===----------------------------------------------------------------------===//

CIM_TEST(brute_force_oracle_agrees_with_hand_computed_cases) {
  // If the oracle is wrong, every test above is meaningless, so it gets its
  // own checks against cases that can be reasoned through by hand.

  // Everything fits: one program per distinct weight.
  CIM_EXPECT_EQ(optimalPrograms({0, 1, 2, 0, 1, 2}, 3), 3u);

  // One tile, alternating: every single access misses.
  CIM_EXPECT_EQ(optimalPrograms({0, 1, 0, 1, 0, 1}, 1), 6u);

  // Two tiles, three weights, cyclic. Optimal keeps two resident and pays
  // for the third each cycle -- strictly better than LRU's total thrash.
  const uint64_t cyclic = optimalPrograms({0, 1, 2, 0, 1, 2, 0, 1, 2}, 2);
  CIM_EXPECT_GE(cyclic, 3u);         // cannot beat one program per weight
  CIM_EXPECT_LT(cyclic, 9u);         // must beat missing on every access

  // Degenerate inputs must not hang or read out of bounds.
  CIM_EXPECT_EQ(optimalPrograms({}, 4), 0u);
  CIM_EXPECT_EQ(optimalPrograms({0, 0, 0}, 0), 0u);
  CIM_EXPECT_EQ(optimalPrograms({7, 7, 7}, 1), 1u);
}

CIM_TEST(the_optimality_test_would_catch_a_degraded_belady) {
  // An optimality check that no realistic wrong answer could fail is not a
  // check. This pins a case where LRU is strictly worse than optimal, so if
  // computePlacement's Belady path ever silently degraded to LRU-like
  // behaviour, belady_is_optimal_* above would fail rather than pass.
  const PlacementProblem cyclic =
      makeProblem({0, 1, 2, 0, 1, 2, 0, 1, 2}, /*numTiles=*/2);

  const uint64_t best = optimalPrograms(cyclic.useSequence, cyclic.numTiles);
  const uint64_t belady = computePlacement(cyclic, EvictionPolicy::Belady).programs;
  const uint64_t lru = computePlacement(cyclic, EvictionPolicy::LRU).programs;

  CIM_EXPECT_EQ(belady, best);
  // LRU thrashes on a cyclic sweep: it evicts precisely the block it is
  // about to need next.
  CIM_EXPECT_GT(lru, best);
}

CIM_TEST(brute_force_oracle_is_never_beaten_by_any_policy) {
  // The oracle claims to be a lower bound. If any real policy ever beats
  // it, the oracle -- not the policy -- is broken.
  runProperty("oracle is a lower bound", 300, 10, 20,
              [](const PlacementProblem &problem, std::string *why) {
                const uint64_t best =
                    optimalPrograms(problem.useSequence, problem.numTiles);
                for (EvictionPolicy policy :
                     {EvictionPolicy::Belady, EvictionPolicy::LRU,
                      EvictionPolicy::FIFO}) {
                  const uint64_t got =
                      computePlacement(problem, policy).programs;
                  if (got < best) {
                    *why = std::string(toString(policy)) + " emitted " +
                           std::to_string(got) +
                           " programs, below the supposed optimum " +
                           std::to_string(best);
                    return false;
                  }
                }
                return true;
              });
}
