//===- CostReportUtils.h - IR-driven op counting for Pass 8 -----*- C++ -*-=//
//
// The counting logic cim-cost-report (spec Sec. 6, Pass 8) runs on the
// final IR, split out from the pass itself so it can be called directly by
// tests without capturing the pass's printed output. See CIMCostReport.cpp
// for why every op is weighted by its enclosing loops' trip counts rather
// than just counted.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_TRANSFORMS_COSTREPORTUTILS_H
#define CIM_TRANSFORMS_COSTREPORTUTILS_H

#include <cstdint>

namespace mlir {
class ModuleOp;

namespace cim {

/// Weighted cim.program / cim.mvm counts recovered from an already-placed
/// module. `programs`/`mvms` are the whole-run totals this pass's JSON
/// report is built from, and the quantity test/mlir/cost_report_e2e_test.cpp
/// checks against cimrt_profile's own counters for the same module actually
/// executed.
struct IRCostCounts {
  /// Sum, over every cim.program site with a fully-known enclosing trip
  /// count, of that site's weight (product of its enclosing loops' trip
  /// counts). A site under a loop with a non-constant trip count is
  /// excluded here -- see unknownProgramSites -- never guessed at.
  uint64_t programs = 0;
  /// Of `programs`' constituent sites, those with zero enclosing loops:
  /// executed exactly once, ever.
  uint64_t installPrograms = 0;
  /// Of `programs`' constituent sites, those with exactly one enclosing
  /// loop of known trip count: one firing per iteration of what this
  /// report treats as "one inference" (v0.1 scope: a single batch loop,
  /// matching cim-placement's own one-level hoisting limit). A site nested
  /// two or more loops deep still counts toward `programs` but is not
  /// classified here.
  uint64_t steadyStatePrograms = 0;
  /// Same weighting as `programs`, for cim.mvm.
  uint64_t mvms = 0;
  /// Same weighting as `programs`, for cim.requantize. No install/steady-
  /// state split here -- that split exists specifically for cim.program's
  /// asymmetric one-time-vs-recurring cost story (spec Sec. 17), which
  /// cim.requantize has no equivalent of: it costs the same whichever
  /// iteration it fires on.
  uint64_t requantizes = 0;
  /// Weighted count of chained adds a cim.reduce_partial site lowers to --
  /// (numPartials - 1) per site, not one per site, matching
  /// lowerReducePartial's own N-1-chained-calls design
  /// (lib/Transforms/CIMLowerToTarget.cpp): an N-operand reduce_partial
  /// costs N-1 cimrt_reduce_add calls, and a single-operand one (never
  /// emitted by cim-partition, but not IR-invalid) costs zero, since it
  /// lowers to a plain forward with no add at all.
  uint64_t reduceAdds = 0;
  /// The same weighted (numOperands - 1) count for cim.reduce_max sites,
  /// kept separate from reduceAdds because the two are charged against
  /// different target-file entries (costs.reduce_max vs
  /// costs.reduce_partial -- a compare-and-select is a different hardware
  /// step from an add; see ReduceMaxCost in include/cim/Target/TargetSpec.h).
  /// A Kh*Kw pooling window is one op site weighted by Kh*Kw-1 calls.
  uint64_t reduceMaxes = 0;
  /// Weighted bytes moved by cim.copy sites that actually cross the
  /// host/device boundary -- the transfers `costs.transfer` describes and
  /// the ones lowerCimCopy routes through cimrt (and so charges).
  ///
  /// WHAT THIS DELIBERATELY DOES NOT INCLUDE, because the omission is
  /// large enough that leaving it undisclosed would be the exact defect
  /// this project refuses:
  ///
  ///   * Implicit staging. lowerProgram writes the whole weight tile
  ///     through cimrt_write before every cimrt_program, and lowerMvm
  ///     stages the activation and reads the result back the same way
  ///     (the interpreter does likewise). None of that traffic is
  ///     represented by a cim.copy op, so no IR walk can see it -- yet
  ///     cimrt's own recordTransfer charges every byte of it. On a small
  ///     placed 8x8 module the implicit share is the MAJORITY of the
  ///     runtime's reported bytes, not a rounding error. cim.copy's own
  ///     docstring ("transfers are always explicit ops, never implicit")
  ///     describes the dialect's intent, not what the lowering currently
  ///     emits.
  ///   * Host-to-host cim.copy. lowerCimCopy short-circuits it to a plain
  ///     memref.copy with no cimrt involvement, so the runtime charges it
  ///     nothing and neither does this. `costs.transfer` is documented as
  ///     a host<->near cost, so this is a scope decision -- but it is
  ///     counted separately below rather than silently folded to zero.
  ///
  /// The consequence for the static-vs-runtime differential
  /// (test/mlir/cost_report_e2e_test.cpp): this figure is NOT expected to
  /// equal cimrt_profile's `bytes`, and that test does not assert it does.
  /// The two executors do not even agree with each other -- the compiled
  /// path hoists staging buffers out of loops while the interpreter
  /// re-stages per op -- so `bytes` is partly an artifact of which
  /// executor ran, which is itself a known gap rather than a property
  /// worth pinning.
  uint64_t transferBytes = 0;
  /// Weighted count of host-to-host cim.copy sites: real data movement
  /// that neither this report nor the runtime charges (see transferBytes).
  /// Surfaced so a reader can tell "no host-to-host copies happened" from
  /// "they happened and cost nothing here".
  uint64_t hostToHostCopies = 0;

  /// Static cim.program sites whose weight could not be determined because
  /// some enclosing loop's trip count is not a compile-time constant.
  uint64_t unknownProgramSites = 0;
  /// Same, for cim.mvm.
  uint64_t unknownMvmSites = 0;
  /// Same, for cim.requantize.
  uint64_t unknownRequantizeSites = 0;
  /// Same, for cim.reduce_partial.
  uint64_t unknownReduceSites = 0;
  /// Same, for cim.copy -- either an unknown enclosing trip count, or a
  /// shape/element type whose byte size is not statically computable.
  uint64_t unknownCopySites = 0;

  /// True iff every site's weight was determined -- `programs`/`mvms`/
  /// `requantizes`/`reduceAdds` are then the exact whole-run totals, not a
  /// lower bound.
  bool complete() const {
    return unknownProgramSites == 0 && unknownMvmSites == 0 &&
           unknownRequantizeSites == 0 && unknownReduceSites == 0 &&
           unknownCopySites == 0;
  }
};

/// Walk `module` and weight every cim.program/cim.mvm by the product of its
/// enclosing scf.for loops' compile-time-constant trip counts.
IRCostCounts countWeightedOps(ModuleOp module);

} // namespace cim
} // namespace mlir

#endif // CIM_TRANSFORMS_COSTREPORTUTILS_H
