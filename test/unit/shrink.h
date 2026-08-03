//===- shrink.h - Counterexample minimization ------------------*- C++ -*-===//
//
// When a property fails on a random 20-element sequence, the raw
// counterexample is nearly useless for debugging. This greedily shrinks it
// while preserving the failure, so what gets reported is a minimal case that
// can be pasted straight into a new CIM_TEST.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_TEST_SHRINK_H
#define CIM_TEST_SHRINK_H

#include "cim/Placement/Placement.h"

#include <functional>
#include <sstream>
#include <string>

namespace cimtest {

/// Returns true if the problem SATISFIES the property (i.e. no bug).
using PlacementProperty =
    std::function<bool(const cim::PlacementProblem &, std::string *why)>;

/// Greedily remove sequence elements, then reduce the tile count, keeping
/// any change that still reproduces the failure.
inline cim::PlacementProblem shrinkCounterexample(cim::PlacementProblem problem,
                                                   const PlacementProperty &prop) {
  std::string ignored;

  bool progress = true;
  while (progress) {
    progress = false;

    // Drop one element at a time; restart the sweep after each success so
    // later elements get a fresh chance once the sequence is shorter.
    for (size_t i = 0; i < problem.useSequence.size(); ++i) {
      cim::PlacementProblem candidate = problem;
      candidate.useSequence.erase(candidate.useSequence.begin() +
                                  static_cast<long>(i));
      if (candidate.useSequence.empty())
        continue;
      if (!prop(candidate, &ignored)) { // still fails => simpler repro
        problem = std::move(candidate);
        progress = true;
        break;
      }
    }
    if (progress)
      continue;

    // Then try shrinking the tile count, which usually makes the eviction
    // decision that triggered the bug much easier to follow by hand.
    if (problem.numTiles > 1) {
      cim::PlacementProblem candidate = problem;
      candidate.numTiles -= 1;
      if (!prop(candidate, &ignored)) {
        problem = std::move(candidate);
        progress = true;
      }
    }
  }
  return problem;
}

/// Render a problem as pasteable C++, so a reported counterexample can
/// become a permanent regression test without retyping.
inline std::string formatProblem(const cim::PlacementProblem &problem) {
  std::ostringstream os;
  os << "numTiles=" << problem.numTiles << " useSequence={";
  for (size_t i = 0; i < problem.useSequence.size(); ++i)
    os << (i ? ", " : "") << problem.useSequence[i];
  os << "}";
  return os.str();
}

} // namespace cimtest

#endif // CIM_TEST_SHRINK_H
