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
  /// Static cim.program sites whose weight could not be determined because
  /// some enclosing loop's trip count is not a compile-time constant.
  uint64_t unknownProgramSites = 0;
  /// Same, for cim.mvm.
  uint64_t unknownMvmSites = 0;

  /// True iff every site's weight was determined -- `programs`/`mvms` are
  /// then the exact whole-run totals, not a lower bound.
  bool complete() const {
    return unknownProgramSites == 0 && unknownMvmSites == 0;
  }
};

/// Walk `module` and weight every cim.program/cim.mvm by the product of its
/// enclosing scf.for loops' compile-time-constant trip counts.
IRCostCounts countWeightedOps(ModuleOp module);

} // namespace cim
} // namespace mlir

#endif // CIM_TRANSFORMS_COSTREPORTUTILS_H
