//===- Placement.cpp - Tile placement/eviction engine ----------*- C++ -*-===//

#include "cim/Placement/Placement.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace cim {

const char *toString(EvictionPolicy policy) {
  switch (policy) {
  case EvictionPolicy::Belady:
    return "belady";
  case EvictionPolicy::LRU:
    return "lru";
  case EvictionPolicy::FIFO:
    return "fifo";
  }
  return "unknown";
}

const char *toString(ActionKind kind) {
  switch (kind) {
  case ActionKind::Reuse:
    return "reuse";
  case ActionKind::ProgramIntoFree:
    return "program";
  case ActionKind::ProgramWithEviction:
    return "program+evict";
  }
  return "unknown";
}

namespace {

/// nextUse[i] = the smallest j > i with seq[j] == seq[i], or kNeverAgain.
/// Computed by sweeping backwards, so this is O(n) rather than O(n^2) — it
/// matters because the use sequence for a real model is one entry per
/// (layer x tile-partition) and gets long quickly.
std::vector<size_t> computeNextUse(const std::vector<WeightId> &seq) {
  std::vector<size_t> nextUse(seq.size(), kNeverAgain);
  std::unordered_map<WeightId, size_t> seenAt;
  for (size_t i = seq.size(); i-- > 0;) {
    auto it = seenAt.find(seq[i]);
    nextUse[i] = (it == seenAt.end()) ? kNeverAgain : it->second;
    seenAt[seq[i]] = i;
  }
  return nextUse;
}

/// Per-tile bookkeeping. Which of these fields drives eviction depends on
/// the policy; all three are maintained unconditionally because the cost is
/// trivial next to keeping three parallel implementations honest.
struct TileState {
  WeightId weight = kNoWeight;
  /// Step index of this weight's next use after the step that made it
  /// resident. kNeverAgain means it is dead weight and is the ideal victim.
  size_t nextUse = kNeverAgain;
  /// Step at which this tile was last touched (LRU).
  size_t lastUsed = 0;
  /// Monotonic counter of when this tile was programmed (FIFO).
  uint64_t programSeq = 0;
};

/// Pick the tile to evict. Every policy here is "score the tiles, take the
/// max", so they differ only in the score.
TileId selectVictim(const std::vector<TileState> &tiles, EvictionPolicy policy) {
  TileId victim = 0;
  switch (policy) {
  case EvictionPolicy::Belady: {
    // Furthest next use wins; kNeverAgain (dead weight) wins outright.
    size_t best = 0;
    bool first = true;
    for (TileId t = 0; t < tiles.size(); ++t) {
      if (first || tiles[t].nextUse > best) {
        best = tiles[t].nextUse;
        victim = t;
        first = false;
      }
      if (best == kNeverAgain)
        break; // cannot do better than evicting a weight that is never used again
    }
    break;
  }
  case EvictionPolicy::LRU: {
    size_t oldest = 0;
    bool first = true;
    for (TileId t = 0; t < tiles.size(); ++t) {
      if (first || tiles[t].lastUsed < oldest) {
        oldest = tiles[t].lastUsed;
        victim = t;
        first = false;
      }
    }
    break;
  }
  case EvictionPolicy::FIFO: {
    uint64_t earliest = 0;
    bool first = true;
    for (TileId t = 0; t < tiles.size(); ++t) {
      if (first || tiles[t].programSeq < earliest) {
        earliest = tiles[t].programSeq;
        victim = t;
        first = false;
      }
    }
    break;
  }
  }
  return victim;
}

} // namespace

PlacementResult computePlacement(const PlacementProblem &problem,
                                  EvictionPolicy policy) {
  PlacementResult result;
  result.policy = policy;

  if (problem.useSequence.empty() || problem.numTiles == 0)
    return result;

  {
    std::unordered_map<WeightId, char> distinct;
    for (WeightId w : problem.useSequence)
      distinct[w] = 1;
    result.distinctWeights = distinct.size();
  }

  const std::vector<size_t> nextUse = computeNextUse(problem.useSequence);

  std::vector<TileState> tiles(problem.numTiles);
  // Which tile currently holds a given weight, so the resident check is O(1).
  std::unordered_map<WeightId, TileId> residency;
  uint32_t tilesInUse = 0;
  uint64_t programCounter = 0;

  result.actions.reserve(problem.useSequence.size());

  for (size_t step = 0; step < problem.useSequence.size(); ++step) {
    const WeightId w = problem.useSequence[step];

    PlacementAction action;
    action.step = step;
    action.weight = w;

    auto resident = residency.find(w);
    if (resident != residency.end()) {
      // Already resident: emit no cim.program at all. This is the win.
      const TileId t = resident->second;
      action.kind = ActionKind::Reuse;
      action.tile = t;
      tiles[t].nextUse = nextUse[step];
      tiles[t].lastUsed = step;
      ++result.reuses;
    } else if (tilesInUse < problem.numTiles) {
      const TileId t = tilesInUse++;
      action.kind = ActionKind::ProgramIntoFree;
      action.tile = t;
      tiles[t].weight = w;
      tiles[t].nextUse = nextUse[step];
      tiles[t].lastUsed = step;
      tiles[t].programSeq = programCounter++;
      residency[w] = t;
      ++result.programs;
    } else {
      const TileId t = selectVictim(tiles, policy);
      const WeightId victim = tiles[t].weight;
      action.kind = ActionKind::ProgramWithEviction;
      action.tile = t;
      action.evicted = victim;
      residency.erase(victim);
      tiles[t].weight = w;
      tiles[t].nextUse = nextUse[step];
      tiles[t].lastUsed = step;
      tiles[t].programSeq = programCounter++;
      residency[w] = t;
      ++result.programs;
      ++result.evictions;
    }

    result.actions.push_back(action);
  }

  return result;
}

bool validatePlacement(const PlacementProblem &problem,
                        const PlacementResult &result, std::string *error) {
  auto fail = [&](const std::string &msg) {
    if (error)
      *error = msg;
    return false;
  };

  if (result.actions.size() != problem.useSequence.size())
    return fail("action count does not match use-sequence length");

  // Replay against a simulated tile array.
  std::vector<WeightId> tileContents(problem.numTiles, kNoWeight);
  uint64_t programs = 0, reuses = 0, evictions = 0;

  for (size_t step = 0; step < result.actions.size(); ++step) {
    const PlacementAction &a = result.actions[step];

    if (a.step != step)
      return fail("action " + std::to_string(step) + " has wrong step index");
    if (a.weight != problem.useSequence[step])
      return fail("action " + std::to_string(step) + " uses the wrong weight");
    if (a.tile >= problem.numTiles)
      return fail("action " + std::to_string(step) + " names tile " +
                  std::to_string(a.tile) + " outside the target's tile count");

    switch (a.kind) {
    case ActionKind::Reuse:
      if (tileContents[a.tile] != a.weight)
        return fail("action " + std::to_string(step) +
                    " claims reuse but tile " + std::to_string(a.tile) +
                    " does not hold that weight");
      ++reuses;
      break;
    case ActionKind::ProgramIntoFree:
      if (tileContents[a.tile] != kNoWeight)
        return fail("action " + std::to_string(step) +
                    " claims a free tile but tile " + std::to_string(a.tile) +
                    " is occupied");
      tileContents[a.tile] = a.weight;
      ++programs;
      break;
    case ActionKind::ProgramWithEviction:
      if (tileContents[a.tile] == kNoWeight)
        return fail("action " + std::to_string(step) +
                    " claims an eviction but tile " + std::to_string(a.tile) +
                    " was free");
      if (tileContents[a.tile] != a.evicted)
        return fail("action " + std::to_string(step) +
                    " reports evicting the wrong weight");
      tileContents[a.tile] = a.weight;
      ++programs;
      ++evictions;
      break;
    }

    // The post-condition that actually matters: after this action, the tile
    // the step names really does hold the weight the step needs.
    if (tileContents[a.tile] != a.weight)
      return fail("after action " + std::to_string(step) + " tile " +
                  std::to_string(a.tile) + " does not hold the required weight");

    // No weight may be resident in two tiles at once — that would let a
    // later step "reuse" a stale copy.
    int copies = 0;
    for (WeightId held : tileContents)
      if (held == a.weight)
        ++copies;
    if (copies != 1)
      return fail("weight is resident in " + std::to_string(copies) +
                  " tiles after action " + std::to_string(step));
  }

  if (programs != result.programs)
    return fail("reported program count does not match the replayed schedule");
  if (reuses != result.reuses)
    return fail("reported reuse count does not match the replayed schedule");
  if (evictions != result.evictions)
    return fail("reported eviction count does not match the replayed schedule");

  return true;
}

SteadyStateResult computeSteadyStatePlacement(const SteadyStateProblem &problem) {
  SteadyStateResult result;
  if (problem.bodySequence.empty() || problem.numTiles == 0)
    return result;
  result.feasible = true;

  // Occurrence count per distinct weight, plus first-use order kept
  // separately: an unordered_map's own iteration order is not
  // deterministic, and two weights used equally often must not have
  // their pin/stream assignment depend on hash-table layout.
  std::unordered_map<WeightId, uint64_t> occurrences;
  std::vector<WeightId> firstUseOrder;
  for (WeightId w : problem.bodySequence) {
    auto [it, inserted] = occurrences.try_emplace(w, uint64_t{0});
    ++it->second;
    if (inserted)
      firstUseOrder.push_back(w);
  }
  const size_t distinctWeights = firstUseOrder.size();

  if (distinctWeights <= problem.numTiles) {
    // Fits entirely: every distinct weight is pinned, nothing streams --
    // the mm-fit case. This re-derives, deliberately, exactly what the
    // pass's own accidental hoisting already reached correctly for this
    // case; the point of this function is the SPILL case below, not a
    // new claim here.
    //
    // Each pinned weight's OWN FIRST occurrence in the body is a real
    // ProgramIntoFree, not a Reuse: the caller's rewrite needs an actual
    // surviving cim.program to hoist for each pinned weight (there is
    // nothing to "reuse" from if every occurrence were labeled Reuse --
    // the op that would have produced the resident value would never
    // exist). Only a weight repeated later in the SAME body reuses that
    // one real install; programsPerIteration stays 0 regardless, since a
    // pinned tile's install is hoisted and never recurs per iteration
    // (see the comment on SteadyStateResult::programsPerIteration).
    std::unordered_map<WeightId, TileId> pinnedTile;
    for (size_t i = 0; i < firstUseOrder.size(); ++i)
      pinnedTile[firstUseOrder[i]] = static_cast<TileId>(i);
    result.pinnedTileCount = static_cast<uint32_t>(distinctWeights);
    result.programsPerIteration = 0;
    result.body.reserve(problem.bodySequence.size());
    std::unordered_set<WeightId> installed;
    for (WeightId w : problem.bodySequence) {
      PlacementAction a;
      a.step = result.body.size();
      a.weight = w;
      a.tile = pinnedTile.at(w);
      a.kind = installed.insert(w).second ? ActionKind::ProgramIntoFree
                                          : ActionKind::Reuse;
      result.body.push_back(a);
    }
    return result;
  }

  // Spill: pin the (numTiles - 1) weights used most often -- ties broken
  // by first-use order, so the choice is deterministic -- and stream
  // everything else through the one tile deliberately left unpinned.
  std::vector<WeightId> byOccurrence = firstUseOrder;
  std::stable_sort(byOccurrence.begin(), byOccurrence.end(),
                    [&](WeightId a, WeightId b) {
                      return occurrences.at(a) > occurrences.at(b);
                    });
  const uint32_t pinnedCount = problem.numTiles - 1;
  std::unordered_map<WeightId, TileId> pinnedTile;
  for (uint32_t i = 0; i < pinnedCount; ++i)
    pinnedTile[byOccurrence[i]] = static_cast<TileId>(i);
  const TileId streamTile = static_cast<TileId>(pinnedCount);
  result.pinnedTileCount = pinnedCount;

  // The streamed subsequence: bodySequence with every pinned occurrence
  // removed, relative order kept. Solved via the ordinary engine with
  // numTiles=1 rather than a second, hand-written scheme -- with a
  // single tile there is never a choice of victim (selectVictim always
  // has exactly one candidate), so which EvictionPolicy is named here is
  // immaterial; Belady is used only because it is the one every other
  // caller already trusts.
  std::vector<WeightId> streamed;
  streamed.reserve(problem.bodySequence.size());
  for (WeightId w : problem.bodySequence)
    if (!pinnedTile.count(w))
      streamed.push_back(w);

  std::vector<PlacementAction> streamActions;
  if (!streamed.empty()) {
    PlacementProblem single;
    single.numTiles = 1;
    single.useSequence = streamed;
    const PlacementResult solved = computePlacement(single, EvictionPolicy::Belady);
    streamActions = solved.actions;
  }

  result.body.reserve(problem.bodySequence.size());
  size_t streamIndex = 0;
  // As in the fits-entirely branch above: a pinned weight's own first
  // occurrence in the body is the real, surviving ProgramIntoFree the
  // caller hoists; only a LATER repeat of that same weight within this
  // body reuses it. programsPerIteration counts only the streamed
  // tile's own programs -- a pinned tile's install, whatever its kind
  // label, is hoisted and never recurs per iteration.
  std::unordered_set<WeightId> installedPinned;
  for (WeightId w : problem.bodySequence) {
    PlacementAction a;
    a.step = result.body.size();
    a.weight = w;
    auto pinnedIt = pinnedTile.find(w);
    if (pinnedIt != pinnedTile.end()) {
      a.tile = pinnedIt->second;
      a.kind = installedPinned.insert(w).second ? ActionKind::ProgramIntoFree
                                                : ActionKind::Reuse;
    } else {
      const PlacementAction &src = streamActions[streamIndex++];
      a.tile = streamTile;
      a.kind = src.kind;
      // The Free/Eviction distinction is bookkeeping this function's own
      // caller never acts on differently (see the header comment on
      // SteadyStateResult::body) -- carried through anyway so a
      // validator working only from `body` has enough information to
      // check it, without needing to re-derive it.
      if (src.kind == ActionKind::ProgramWithEviction)
        a.evicted = src.evicted;
      if (src.kind != ActionKind::Reuse)
        ++result.programsPerIteration;
    }
    result.body.push_back(a);
  }

  return result;
}

bool validateSteadyStatePlacement(const SteadyStateProblem &problem,
                                   const SteadyStateResult &result,
                                   std::string *error) {
  auto fail = [&](const std::string &msg) {
    if (error)
      *error = msg;
    return false;
  };

  if (!result.feasible)
    return fail("steady-state result is not feasible");
  if (result.body.size() != problem.bodySequence.size())
    return fail("steady-state body length does not match the use sequence");
  for (size_t i = 0; i < result.body.size(); ++i)
    if (result.body[i].weight != problem.bodySequence[i])
      return fail("steady-state body step " + std::to_string(i) +
                  " names a different weight than the use sequence");

  // The stream tile's "continuation" actions: what a SECOND (and every
  // later) repetition's stream-tile steps look like, now that the tile
  // is no longer empty. Recomputed here rather than stored on
  // SteadyStateResult -- the real IR rewrite in
  // lib/Transforms/CIMPlacement.cpp never needs it, since it treats
  // ProgramIntoFree and ProgramWithEviction identically; only this proof
  // does, to satisfy validatePlacement's own bookkeeping.
  std::vector<WeightId> streamed;
  for (const PlacementAction &a : result.body)
    if (a.tile >= result.pinnedTileCount)
      streamed.push_back(a.weight);

  std::vector<PlacementAction> continuation;
  if (!streamed.empty()) {
    PlacementProblem doubled;
    doubled.numTiles = 1;
    doubled.useSequence = streamed;
    doubled.useSequence.insert(doubled.useSequence.end(), streamed.begin(),
                               streamed.end());
    const PlacementResult solved =
        computePlacement(doubled, EvictionPolicy::Belady);
    continuation.assign(solved.actions.begin() + streamed.size(),
                        solved.actions.end());
    // The internal solve above used a single tile numbered 0 (it knows
    // nothing about the real target's tile count); remap every action to
    // the ACTUAL physical stream tile before this is used in a real,
    // full-width replay below -- `computeSteadyStatePlacement` performs
    // the identical remap for `result.body` itself, for the same reason.
    for (PlacementAction &a : continuation)
      a.tile = result.pinnedTileCount;
  }

  PlacementProblem flat;
  flat.numTiles = problem.numTiles;
  flat.name = problem.name;
  PlacementResult flatResult;
  flatResult.policy = EvictionPolicy::Belady;

  auto append = [&](WeightId w, const PlacementAction &action) {
    PlacementAction a = action;
    a.step = flat.useSequence.size();
    a.weight = w;
    flat.useSequence.push_back(w);
    flatResult.actions.push_back(a);
    if (a.kind == ActionKind::Reuse) {
      ++flatResult.reuses;
    } else {
      ++flatResult.programs;
      if (a.kind == ActionKind::ProgramWithEviction)
        ++flatResult.evictions;
    }
  };

  // No separate bootstrap pass is needed: computeSteadyStatePlacement
  // already guarantees that a pinned weight's FIRST occurrence in `body`
  // is a real ActionKind::ProgramIntoFree (not a Reuse) at whatever step
  // it naturally falls, and every later occurrence of that same weight is
  // Reuse -- so replaying `body` verbatim once already installs every
  // pinned tile in place, in the correct order, against an initially
  // empty device. That is exactly repetition 0 below.
  //
  // Three repetitions of `body`: repetition 0 uses `body`'s own actions
  // unchanged, which both installs every pinned tile (its first
  // ProgramIntoFree) and streams the first pass through the stream tile
  // (also starting empty). Repetitions 1 and 2 force EVERY pinned-tile
  // step to Reuse regardless of what `body` itself recorded there --  by
  // then the tile is permanently occupied by that same weight forever, so
  // replaying body's own first-occurrence ProgramIntoFree again would
  // fail validatePlacement's own free-tile check -- and use
  // `continuation`'s recomputed labels for the stream tile, since it is
  // no longer empty either by the second pass.
  for (int rep = 0; rep < 3; ++rep) {
    size_t streamIndex = 0;
    for (const PlacementAction &a : result.body) {
      if (a.tile < result.pinnedTileCount) {
        if (rep == 0) {
          append(a.weight, a);
        } else {
          PlacementAction reuse = a;
          reuse.kind = ActionKind::Reuse;
          append(a.weight, reuse);
        }
      } else {
        const PlacementAction &src = (rep == 0) ? a : continuation[streamIndex];
        append(a.weight, src);
        ++streamIndex;
      }
    }
  }

  return validatePlacement(flat, flatResult, error);
}

} // namespace cim
