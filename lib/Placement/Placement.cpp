//===- Placement.cpp - Tile placement/eviction engine ----------*- C++ -*-===//

#include "cim/Placement/Placement.h"

#include <algorithm>
#include <unordered_map>

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

} // namespace cim
