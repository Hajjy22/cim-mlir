//===- LoopAnalysis.h - Shared scf.for trip-count queries -------*- C++ -*-=//
//
// Extracted from cim-placement (spec Sec. 6, Pass 3), which needs "does this
// loop provably run at least once" for its hoisting decision, and shared
// with cim-cost-report (spec Sec. 6, Pass 8), which needs the exact trip
// count to weight ops inside a loop -- a hoisted cim.program executes once
// regardless of position, but an op still inside the loop body executes
// trip-count times, so a cost report that does not know the trip count
// cannot report a runtime-accurate number. One implementation, so the two
// passes' notion of "provably constant" never drifts apart.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_TRANSFORMS_LOOPANALYSIS_H
#define CIM_TRANSFORMS_LOOPANALYSIS_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace scf {
class ForOp;
} // namespace scf

namespace cim {

/// The exact number of iterations `forOp` executes, or failure if that
/// cannot be determined at compile time (a non-constant bound/step, or a
/// non-positive step -- scf.for's semantics do not guarantee termination
/// for those, so guessing a count would be exactly the "silently produce a
/// wrong-but-plausible number" this project's passes refuse to do).
///
/// A loop whose constant bounds prove it never executes (lower >= upper for
/// a positive step) returns 0 -- that is a real, exact answer, not a
/// failure to compute one.
FailureOr<int64_t> getConstantTripCount(scf::ForOp forOp);

/// True iff `forOp` is provably going to execute at least one iteration:
/// bounds and step are all compile-time constants and lower < upper (step
/// already assumed forward per scf.for's semantics; a non-positive step is
/// treated as unprovable rather than reasoned about further).
///
/// Equivalent to `getConstantTripCount(forOp)` succeeding with a value > 0;
/// kept as its own query because cim-placement's hoisting check only needs
/// the boolean, not the count.
bool loopHasProvablyPositiveTripCount(scf::ForOp forOp);

} // namespace cim
} // namespace mlir

#endif // CIM_TRANSFORMS_LOOPANALYSIS_H
