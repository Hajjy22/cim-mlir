//===- validator_mutation_test.cpp - Mutation-test the validator *- C++ -*-===//
//
// validatePlacement is what every "correct" number in cim-bench rests on:
// each schedule is replayed against a simulated tile array before its
// program count is reported. A validator that never actually fails would
// make all of those numbers meaningless while looking exactly the same.
//
// This is a mutation test. Each row corrupts a valid schedule in one
// specific way and asserts the validator both rejects it and says why. Any
// row that starts passing silently means a branch has stopped working.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Placement/Placement.h"

#include <functional>
#include <string>
#include <vector>

using namespace cim;

namespace {

/// A schedule that fits comfortably and exercises reuse, programs into free
/// tiles, and eviction -- so every mutation below has something to corrupt.
PlacementProblem baseProblem() {
  PlacementProblem problem;
  problem.numTiles = 2;
  problem.useSequence = {0, 1, 0, 2, 1, 2, 0};
  problem.name = "mutation-base";
  return problem;
}

struct Mutation {
  const char *name;
  /// Corrupt the result in place. Returns false if this schedule has no
  /// action of the kind the mutation needs, so the row can be skipped
  /// rather than silently passing.
  std::function<bool(PlacementResult &)> apply;
  const char *expectedMessage;
};

/// Index of the first action of a given kind, or -1.
long findAction(const PlacementResult &r, ActionKind kind) {
  for (size_t i = 0; i < r.actions.size(); ++i)
    if (r.actions[i].kind == kind)
      return static_cast<long>(i);
  return -1;
}

} // namespace

CIM_TEST(validator_accepts_the_unmutated_schedule) {
  // The control. If this ever fails, every rejection below proves nothing.
  const PlacementProblem problem = baseProblem();
  const PlacementResult result =
      computePlacement(problem, EvictionPolicy::Belady);
  std::string error;
  CIM_EXPECT(validatePlacement(problem, result, &error));
  CIM_EXPECT_EQ(error, std::string(""));
}

CIM_TEST(validator_rejects_every_corruption) {
  const PlacementProblem problem = baseProblem();
  const PlacementResult clean =
      computePlacement(problem, EvictionPolicy::Belady);

  // Sanity: the base schedule must actually contain each action kind, or
  // the corresponding mutations below would be vacuous.
  CIM_EXPECT_GE(findAction(clean, ActionKind::Reuse), 0);
  CIM_EXPECT_GE(findAction(clean, ActionKind::ProgramIntoFree), 0);
  CIM_EXPECT_GE(findAction(clean, ActionKind::ProgramWithEviction), 0);

  const std::vector<Mutation> mutations = {
      {"truncated action list",
       [](PlacementResult &r) {
         r.actions.pop_back();
         return true;
       },
       "action count"},

      {"wrong step index",
       [](PlacementResult &r) {
         r.actions[1].step = 99;
         return true;
       },
       "wrong step index"},

      {"wrong weight",
       [](PlacementResult &r) {
         r.actions[1].weight = 12345;
         return true;
       },
       "wrong weight"},

      {"tile out of range",
       [&](PlacementResult &r) {
         r.actions[1].tile = problem.numTiles;
         return true;
       },
       "outside the target's tile count"},

      {"reuse of a tile that does not hold the weight",
       [&](PlacementResult &r) {
         const long i = findAction(r, ActionKind::Reuse);
         if (i < 0)
           return false;
         r.actions[i].tile = (r.actions[i].tile + 1) % problem.numTiles;
         return true;
       },
       "does not hold"},

      {"program claims a free tile that is occupied",
       [](PlacementResult &r) {
         const long i = findAction(r, ActionKind::ProgramWithEviction);
         if (i < 0)
           return false;
         r.actions[i].kind = ActionKind::ProgramIntoFree;
         return true;
       },
       "claims a free tile"},

      {"eviction claimed on a free tile",
       [](PlacementResult &r) {
         const long i = findAction(r, ActionKind::ProgramIntoFree);
         if (i < 0)
           return false;
         r.actions[i].kind = ActionKind::ProgramWithEviction;
         r.actions[i].evicted = 0;
         return true;
       },
       "claims an eviction"},

      {"wrong evicted weight",
       [](PlacementResult &r) {
         const long i = findAction(r, ActionKind::ProgramWithEviction);
         if (i < 0)
           return false;
         r.actions[i].evicted = 4242;
         return true;
       },
       "evicting the wrong weight"},

      {"program count under-reported",
       [](PlacementResult &r) {
         if (r.programs == 0)
           return false;
         r.programs -= 1;
         return true;
       },
       "program count"},

      {"reuse count over-reported",
       [](PlacementResult &r) {
         r.reuses += 1;
         return true;
       },
       "reuse count"},

      {"eviction count over-reported",
       [](PlacementResult &r) {
         r.evictions += 1;
         return true;
       },
       "eviction count"},
  };

  for (const Mutation &mutation : mutations) {
    PlacementResult corrupted = clean;
    if (!mutation.apply(corrupted)) {
      CIM_FAIL(std::string("mutation '") + mutation.name +
               "' could not be applied: the base schedule lacks the action "
               "kind it needs, so this row proves nothing");
      continue;
    }

    std::string error;
    if (validatePlacement(problem, corrupted, &error)) {
      CIM_FAIL(std::string("validator ACCEPTED a corrupted schedule: ") +
               mutation.name);
      continue;
    }
    CIM_EXPECT_CONTAINS(error, mutation.expectedMessage);
  }
}

CIM_TEST(validator_rejects_a_weight_resident_in_two_tiles) {
  // A duplicated residency would let a later step "reuse" a stale copy,
  // which is the failure mode SSA residency (spec Sec. 5.4 rule 3) exists
  // to prevent.
  PlacementProblem problem;
  problem.numTiles = 2;
  problem.useSequence = {0, 1, 0};

  PlacementResult forged;
  forged.programs = 2;
  forged.reuses = 1;
  forged.evictions = 0;
  forged.distinctWeights = 2;
  forged.actions = {
      {0, /*weight=*/0, /*tile=*/0, ActionKind::ProgramIntoFree, kNoWeight},
      // Claims weight 0 again, into the other tile, as a fresh program.
      {1, /*weight=*/0, /*tile=*/1, ActionKind::ProgramIntoFree, kNoWeight},
      {2, /*weight=*/0, /*tile=*/0, ActionKind::Reuse, kNoWeight},
  };
  forged.actions[1].weight = 0;

  std::string error;
  CIM_EXPECT(!validatePlacement(problem, forged, &error));
}
