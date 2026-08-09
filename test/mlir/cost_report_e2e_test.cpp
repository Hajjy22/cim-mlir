//===- cost_report_e2e_test.cpp - Static vs runtime cost differential ---===//
//
// cim-cost-report (spec Sec. 6, Pass 8) predicts programs/mvms statically,
// weighting every op by its enclosing loops' trip counts (see
// cim/Transforms/CostReportUtils.h for why a plain walk-and-count would be
// wrong on exactly the IR cim-placement's loop hoisting now produces).
//
// This is the strong check the plan calls for: cim-run --profile already
// prints REAL programs/mvms counters from cimrt after actually executing a
// module, trip counts and all. So for any module, cim-cost-report's
// predicted counts must equal what actually executes -- a genuine
// differential between two independently-derived numbers, in the same
// spirit as the numpy and PyYAML differentials elsewhere in this project.
// The loop cases below are the ones that fail if trip-count weighting is
// missing (i.e. a naive static op count would under-report programs/mvms
// by exactly the trip count).
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Dialect/CIMDialect.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Interpreter/Interpreter.h"
#include "cim/Transforms/CostReportUtils.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace mlir;

namespace {

std::string tinyTarget() {
  static std::string path = [] {
    for (const char *candidate :
         {"test/targets/tiny-4x4.yaml", "../test/targets/tiny-4x4.yaml",
          "../../test/targets/tiny-4x4.yaml"}) {
      if (FILE *f = std::fopen(candidate, "r")) {
        std::fclose(f);
        return std::string(candidate);
      }
    }
    return std::string("test/targets/tiny-4x4.yaml");
  }();
  return path;
}

/// Runtime counters actually parsed out of `cim-run --profile`'s
/// "cimrt_profile programs=N mvms=M bytes=B" line, so the comparison below
/// is against cimrt's own accounting, not a re-derivation of it.
struct RuntimeCounts {
  uint64_t programs = 0;
  uint64_t mvms = 0;
  uint64_t requantizes = 0;
  bool found = false;
};

RuntimeCounts parseRuntimeCounts(const std::string &text) {
  RuntimeCounts counts;
  const std::string marker = "cimrt_profile programs=";
  const size_t pos = text.find(marker);
  if (pos == std::string::npos)
    return counts;
  counts.found = true;
  counts.programs = static_cast<uint64_t>(
      std::strtoull(text.c_str() + pos + marker.size(), nullptr, 10));
  const size_t mvmsPos = text.find("mvms=", pos);
  counts.mvms = static_cast<uint64_t>(
      std::strtoull(text.c_str() + mvmsPos + 5, nullptr, 10));
  const size_t requantizesPos = text.find("requantizes=", pos);
  counts.requantizes = static_cast<uint64_t>(std::strtoull(
      text.c_str() + requantizesPos + std::strlen("requantizes="), nullptr,
      10));
  return counts;
}

/// Compile `source` through cim-detect + cim-partition (+ cim-placement
/// when `withPlacement`), predict programs/mvms via countWeightedOps, then
/// actually execute the SAME module through the interpreter with profiling
/// on, and return both so the caller can assert they agree.
struct Differential {
  cim::IRCostCounts predicted;
  RuntimeCounts actual;
  std::string error;
};

Differential runDifferential(const std::string &source, bool withPlacement,
                             bool withLegalizePrecision = false) {
  Differential result;

  DialectRegistry registry;
  registry.insert<cim::CIMDialect, func::FuncDialect, memref::MemRefDialect,
                  arith::ArithDialect, linalg::LinalgDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  if (!module) {
    result.error = "failed to parse the generated module";
    return result;
  }

  PassManager pm(&context);
  pm.addPass(cim::createCIMDetectPass());
  auto partition = cim::createCIMPartitionPass();
  if (failed(partition->initializeOptions("target-yaml=" + tinyTarget()))) {
    result.error = "failed to set cim-partition options";
    return result;
  }
  pm.addPass(std::move(partition));
  if (withPlacement) {
    auto placement = cim::createCIMPlacementPass();
    if (failed(placement->initializeOptions("target-yaml=" + tinyTarget()))) {
      result.error = "failed to set cim-placement options";
      return result;
    }
    pm.addPass(std::move(placement));
  }
  if (withLegalizePrecision) {
    // The one place this differential exercises cim.requantize: added
    // directly after partition(+placement), the minimal pipeline that
    // gets one into the IR, matching this file's existing minimal-pipeline
    // style (no cim-schedule/cim-insert-transfers here either -- the
    // interpreter needs neither to execute cim.requantize correctly, and
    // this test is about the count, not about composition).
    auto legalize = cim::createCIMLegalizePrecisionPass();
    if (failed(legalize->initializeOptions("target-yaml=" + tinyTarget()))) {
      result.error = "failed to set cim-legalize-precision options";
      return result;
    }
    pm.addPass(std::move(legalize));
  }
  if (failed(pm.run(*module))) {
    result.error = "pass pipeline failed";
    return result;
  }

  // Predict from the final, already-placed IR -- exactly what
  // cim-cost-report does.
  result.predicted = cim::countWeightedOps(*module);

  // Actually run it and read cimrt's own counters.
  std::string captured;
  llvm::raw_string_ostream stream(captured);
  cim::InterpreterOptions options;
  options.targetYAMLPath = tinyTarget();
  options.out = &stream;
  options.dumpProfile = true;
  if (failed(cim::run(*module, "main", options))) {
    result.error = "interpreter failed to execute the compiled module";
    return result;
  }
  stream.flush();
  result.actual = parseRuntimeCounts(captured);
  if (!result.actual.found)
    result.error = "no cimrt_profile line in interpreter output: " + captured;
  return result;
}

/// Straight-line: one 4x4 matmul, fits in a single tile. No loop, so every
/// op has weight 1 -- the baseline case where a naive static count would
/// already be right, kept so the differential test also covers "nothing
/// interesting happens" rather than only loop cases.
const char *kStraightLineFits = R"mlir(
memref.global "private" constant @w : memref<4x4xi8> = dense<1>
memref.global "private" constant @a : memref<1x4xi8> = dense<2>
func.func private @cim_print_i32(memref<*xi32>)
func.func @main() {
  %w = memref.get_global @w : memref<4x4xi8>
  %aInit = memref.get_global @a : memref<1x4xi8>
  %a = memref.alloc() : memref<1x4xi8>
  memref.copy %aInit, %a : memref<1x4xi8> to memref<1x4xi8>
  %out = memref.alloc() : memref<1x4xi32>
  linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<4x4xi8>)
    outs(%out : memref<1x4xi32>)
  %u = memref.cast %out : memref<1x4xi32> to memref<*xi32>
  func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()
  return
}
)mlir";

/// Straight-line spill: 12x4 weights over 4x4 tiles is 3 distinct blocks on
/// a 2-tile device -- forces reprogramming with no loop involved, so this
/// exercises the "programs > install-only" arithmetic without also
/// involving trip-count weighting.
const char *kStraightLineSpill = R"mlir(
memref.global "private" constant @w : memref<12x4xi8> = dense<1>
memref.global "private" constant @a : memref<1x4xi8> = dense<2>
func.func private @cim_print_i32(memref<*xi32>)
func.func @main() {
  %w = memref.get_global @w : memref<12x4xi8>
  %aInit = memref.get_global @a : memref<1x4xi8>
  %a = memref.alloc() : memref<1x4xi8>
  memref.copy %aInit, %a : memref<1x4xi8> to memref<1x4xi8>
  %out = memref.alloc() : memref<1x12xi32>
  linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<12x4xi8>)
    outs(%out : memref<1x12xi32>)
  %u = memref.cast %out : memref<1x12xi32> to memref<*xi32>
  func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()
  return
}
)mlir";

/// One weight block, loop-invariant across 3 iterations. Without placement
/// the single textual cim.program still fires on every runtime iteration
/// (weight 3); with placement it is hoisted above the loop (weight 1). This
/// is THE case that fails if trip-count weighting is missing: a plain
/// static count sees exactly one cim.program textually either way and
/// would report "1" for both runs, which is right for the placed case and
/// wrong -- by a factor of 3 -- for the unplaced one.
const char *kLoopFits = R"mlir(
memref.global "private" constant @w : memref<4x4xi8> = dense<[
  [ 1,  0,  0,  0],
  [ 0,  1,  0,  0],
  [ 0,  0,  1,  0],
  [ 0,  0,  0,  1]]>
memref.global "private" constant @acts : memref<3x4xi8> = dense<[
  [ 1,  2,  3,  4],
  [10, 20, 30, 40],
  [-1, -2, -3, -4]]>
func.func private @cim_print_i32(memref<*xi32>)
func.func @main() {
  %w = memref.get_global @w : memref<4x4xi8>
  %acts = memref.get_global @acts : memref<3x4xi8>
  %out = memref.alloc() : memref<3x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    %actRow = memref.subview %acts[%i, 0] [1, 4] [1, 1]
      : memref<3x4xi8> to memref<1x4xi8, strided<[4, 1], offset: ?>>
    %actLocal = memref.alloc() : memref<1x4xi8>
    memref.copy %actRow, %actLocal
      : memref<1x4xi8, strided<[4, 1], offset: ?>> to memref<1x4xi8>
    %outRow = memref.subview %out[%i, 0] [1, 4] [1, 1]
      : memref<3x4xi32> to memref<1x4xi32, strided<[4, 1], offset: ?>>
    linalg.matmul_transpose_b ins(%actLocal, %w : memref<1x4xi8>, memref<4x4xi8>)
      outs(%outRow : memref<1x4xi32, strided<[4, 1], offset: ?>>)
    memref.dealloc %actLocal : memref<1x4xi8>
  }
  %cast = memref.cast %out : memref<3x4xi32> to memref<*xi32>
  func.call @cim_print_i32(%cast) : (memref<*xi32>) -> ()
  memref.dealloc %out : memref<3x4xi32>
  return
}
)mlir";

/// 12x4 weights (3 blocks) over 2 tiles, inside a 2-iteration loop: one
/// block is stable and hoists (weight 1), the other two are reprogrammed
/// every iteration (weight 2 each) -- a mix of install-time and
/// steady-state cost in the SAME module, which is what the install/
/// steady-state split needs to get right, not just the total.
const char *kLoopSpill = R"mlir(
memref.global "private" constant @w2 : memref<12x4xi8> = dense<1>
memref.global "private" constant @acts2 : memref<2x4xi8> = dense<2>
func.func private @cim_print_i32(memref<*xi32>)
func.func @main() {
  %w = memref.get_global @w2 : memref<12x4xi8>
  %acts = memref.get_global @acts2 : memref<2x4xi8>
  %out = memref.alloc() : memref<2x12xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    %actRow = memref.subview %acts[%i, 0] [1, 4] [1, 1]
      : memref<2x4xi8> to memref<1x4xi8, strided<[4, 1], offset: ?>>
    %actLocal = memref.alloc() : memref<1x4xi8>
    memref.copy %actRow, %actLocal
      : memref<1x4xi8, strided<[4, 1], offset: ?>> to memref<1x4xi8>
    %outRow = memref.subview %out[%i, 0] [1, 12] [1, 1]
      : memref<2x12xi32> to memref<1x12xi32, strided<[12, 1], offset: ?>>
    linalg.matmul_transpose_b ins(%actLocal, %w : memref<1x4xi8>, memref<12x4xi8>)
      outs(%outRow : memref<1x12xi32, strided<[12, 1], offset: ?>>)
    memref.dealloc %actLocal : memref<1x4xi8>
  }
  %cast = memref.cast %out : memref<2x12xi32> to memref<*xi32>
  func.call @cim_print_i32(%cast) : (memref<*xi32>) -> ()
  memref.dealloc %out : memref<2x12xi32>
  return
}
)mlir";

void expectDifferentialAgrees(const char *label, const std::string &source,
                              bool withPlacement,
                              bool withLegalizePrecision = false) {
  Differential d = runDifferential(source, withPlacement,
                                   withLegalizePrecision);
  if (!d.error.empty()) {
    CIM_FAIL(std::string(label) + ": " + d.error);
    return;
  }
  CIM_EXPECT_EQ(d.predicted.programs, d.actual.programs);
  CIM_EXPECT_EQ(d.predicted.mvms, d.actual.mvms);
  // Trivially 0 == 0 whenever withLegalizePrecision is false (no
  // cim.requantize is ever emitted without it) -- asserted unconditionally
  // anyway so every existing case here doubles as a "still zero" check,
  // and cost_report_matches_runtime_on_a_requantized_module below is the
  // one that makes it nonzero on both sides.
  CIM_EXPECT_EQ(d.predicted.requantizes, d.actual.requantizes);
  CIM_EXPECT(d.predicted.complete());
}

} // namespace

CIM_TEST(cost_report_matches_runtime_on_straight_line_fitting_module) {
  expectDifferentialAgrees("straight-line-fits", kStraightLineFits,
                           /*withPlacement=*/true);
}

CIM_TEST(cost_report_matches_runtime_on_straight_line_spilling_module) {
  expectDifferentialAgrees("straight-line-spill", kStraightLineSpill,
                           /*withPlacement=*/true);
}

CIM_TEST(cost_report_matches_runtime_on_a_hoisted_loop) {
  // The case that fails without trip-count weighting: a static walk sees
  // one cim.program either way, but the placed run truly executes it once
  // and the unplaced run truly executes it three times.
  expectDifferentialAgrees("loop-fits-placed", kLoopFits,
                           /*withPlacement=*/true);
  expectDifferentialAgrees("loop-fits-unplaced", kLoopFits,
                           /*withPlacement=*/false);
}

CIM_TEST(cost_report_matches_runtime_on_a_partially_hoisted_loop) {
  expectDifferentialAgrees("loop-spill-placed", kLoopSpill,
                           /*withPlacement=*/true);
  expectDifferentialAgrees("loop-spill-unplaced", kLoopSpill,
                           /*withPlacement=*/false);
}

CIM_TEST(cost_report_matches_runtime_on_a_requantized_module) {
  // Every case above runs detect+partition(+placement) only, so none of
  // them ever emit a cim.requantize -- predicted.requantizes and
  // actual.requantizes are both trivially 0 either way, which proves
  // nothing about whether the two sides would actually agree on a REAL
  // count. This is that proof: cim-legalize-precision inserts exactly one
  // cim.requantize (a single 4x4 matmul, one terminal accumulator), and
  // after this session's runtime-side fix (cimrt_requantize now calls
  // CostAccumulator::recordRequantize, see runtime/src/simulator/
  // cost_model.h) the interpreter's own cimrt_profile_stop counts it --
  // so the static prediction and the actually-executed count have
  // something nonzero to agree on for the first time.
  expectDifferentialAgrees("requantized", kStraightLineFits,
                           /*withPlacement=*/true,
                           /*withLegalizePrecision=*/true);
}

CIM_TEST(cost_report_install_vs_steady_state_reflects_hoisting) {
  // Beyond the totals: install should be exactly the hoisted program (paid
  // once), and steady-state-per-inference should be exactly the two that
  // stayed in the loop (paid every iteration) -- the whole point of the
  // spec Sec.17 install/steady-state split.
  Differential placed = runDifferential(kLoopSpill, /*withPlacement=*/true);
  if (!placed.error.empty()) {
    CIM_FAIL(std::string("loop-spill install/steady: ") + placed.error);
    return;
  }
  CIM_EXPECT_EQ(placed.predicted.installPrograms, 1u);
  CIM_EXPECT_EQ(placed.predicted.steadyStatePrograms, 2u);
}
