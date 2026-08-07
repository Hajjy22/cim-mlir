//===- schedule_e2e_test.cpp - cim-schedule numerical invariance --------===//
//
// cim-schedule (spec Sec. 6, Pass 4) only inserts cim.barrier -- a no-op in
// the functional simulator (cimrt_barrier: "executes synchronously; nothing
// to wait on", runtime/src/simulator/simulator.cpp). So the one thing this
// file checks is exactly the thing test/Transforms/cim-schedule.mlir
// CANNOT: that running the SAME module with and without cim-schedule
// produces IDENTICAL numbers. The structural question -- are the barriers
// in the right place -- is what cim-schedule.mlir answers instead; a
// numerical differential could not tell a correctly-placed barrier from a
// missing one, since neither changes what gets computed.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Dialect/CIMDialect.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Interpreter/Interpreter.h"
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

struct RunResult {
  std::string printed;
  unsigned barriers = 0;
  std::string error;
};

/// Compile `source` through cim-detect + cim-partition + cim-placement, plus
/// cim-schedule when `withSchedule`, and execute the result. Running the
/// exact same source both ways is what makes the comparison meaningful:
/// cim-schedule is the only difference, so any difference in printed output
/// is a bug in cim-schedule by definition.
RunResult run(const std::string &source, bool withSchedule) {
  RunResult result;

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
  auto placement = cim::createCIMPlacementPass();
  if (failed(placement->initializeOptions("target-yaml=" + tinyTarget()))) {
    result.error = "failed to set cim-placement options";
    return result;
  }
  pm.addPass(std::move(placement));
  if (withSchedule)
    pm.addPass(cim::createCIMSchedulePass());
  if (failed(pm.run(*module))) {
    result.error = "pass pipeline failed";
    return result;
  }

  module->walk([&](cim::BarrierOp) { ++result.barriers; });

  std::string captured;
  llvm::raw_string_ostream stream(captured);
  cim::InterpreterOptions options;
  options.targetYAMLPath = tinyTarget();
  options.out = &stream;
  if (failed(cim::run(*module, "main", options))) {
    result.error = "interpreter failed to execute the compiled module";
    return result;
  }
  stream.flush();
  result.printed = captured;
  return result;
}

void expectInvariant(const char *label, const std::string &source) {
  RunResult without = run(source, /*withSchedule=*/false);
  if (!without.error.empty()) {
    CIM_FAIL(std::string(label) + " (no schedule): " + without.error);
    return;
  }
  RunResult with = run(source, /*withSchedule=*/true);
  if (!with.error.empty()) {
    CIM_FAIL(std::string(label) + " (scheduled): " + with.error);
    return;
  }
  // The point of the pass: it must actually have inserted at least one
  // barrier, or this whole comparison would trivially pass by doing
  // nothing.
  CIM_EXPECT(with.barriers > 0);
  CIM_EXPECT_EQ(without.printed, with.printed);
}

const char *kStraightLine = R"mlir(
memref.global "private" constant @w : memref<4x4xi8> = dense<[
  [ 1,  0,  0,  0],
  [ 0,  1,  0,  0],
  [ 0,  0,  1,  0],
  [ 0,  0,  0,  1]]>
memref.global "private" constant @a : memref<1x4xi8> = dense<[[5, -3, 100, -128]]>
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

/// 12x4 weights (3 blocks) over 2 tiles: forces reprogramming with no loop
/// involved, exercising cim-schedule's "consecutive same-device work" and
/// "device changes underneath it" bookkeeping (tile_alloc between programs)
/// on a case that is not simply one program then one mvm.
const char *kSpill = R"mlir(
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

/// Loop-hoisting shape: cim-placement hoists the program above the loop, so
/// scheduling this exercises the case that motivated dependsOnAny's nested
/// walk -- a barrier belongs before the loop AND once per iteration inside
/// it, and neither placement is optional for correctness.
const char *kLoop = R"mlir(
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

} // namespace

CIM_TEST(cim_schedule_does_not_change_a_straight_line_result) {
  expectInvariant("straight-line", kStraightLine);
}

CIM_TEST(cim_schedule_does_not_change_a_spill_result) {
  expectInvariant("spill", kSpill);
}

CIM_TEST(cim_schedule_does_not_change_a_hoisted_loop_result) {
  expectInvariant("loop", kLoop);
}
