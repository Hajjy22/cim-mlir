//===- insert_transfers_e2e_test.cpp - cim-insert-transfers invariance --===//
//
// cim-insert-transfers (spec Sec. 6, Pass 5) only inserts cim.copy -- a
// faithful byte-for-byte move between address spaces, not a value-changing
// operation. The interpreter's runMvm already stages an activation into
// near-space device memory internally regardless of the memref's own
// declared space (see Interpreter.cpp), so the one thing this file checks
// is that running the SAME module with and without cim-insert-transfers
// computes IDENTICAL numbers -- the same shape of check as
// schedule_e2e_test.cpp for cim.barrier, for the same reason: cim-partition
// already stages every activation into near space itself
// (CIMPartition.cpp), so this pass has nothing to do on real pipeline
// output and is exercised here on hand-written IR instead, the same
// isolation test/Transforms/cim-insert-transfers.mlir uses.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Dialect/CIMDialect.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Interpreter/Interpreter.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  unsigned copies = 0;
  std::string error;
};

/// Parse `source` (hand-written cim IR, not run through cim-detect or
/// cim-partition -- see the file header for why) and execute it, with
/// cim-insert-transfers in the pipeline when `withPass`. Running the exact
/// same source both ways is what makes the comparison meaningful: the pass
/// is the only difference, so any difference in printed output is a bug in
/// it by definition.
RunResult run(const std::string &source, bool withPass) {
  RunResult result;

  DialectRegistry registry;
  registry.insert<cim::CIMDialect, func::FuncDialect, memref::MemRefDialect,
                  arith::ArithDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  if (!module) {
    result.error = "failed to parse the generated module";
    return result;
  }

  if (withPass) {
    PassManager pm(&context);
    pm.addPass(cim::createCIMInsertTransfersPass());
    if (failed(pm.run(*module))) {
      result.error = "cim-insert-transfers failed";
      return result;
    }
  }

  module->walk([&](cim::CopyOp) { ++result.copies; });

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
  RunResult without = run(source, /*withPass=*/false);
  if (!without.error.empty()) {
    CIM_FAIL(std::string(label) + " (no pass): " + without.error);
    return;
  }
  RunResult with = run(source, /*withPass=*/true);
  if (!with.error.empty()) {
    CIM_FAIL(std::string(label) + " (with pass): " + with.error);
    return;
  }
  // The point of the pass: it must actually have inserted at least one
  // cim.copy, or this whole comparison would trivially pass by doing
  // nothing.
  CIM_EXPECT(with.copies > 0);
  CIM_EXPECT_EQ(without.printed, with.printed);
}

// A single mvm reading a host-space activation directly (no
// #cim.space<near>, no wrapping loop): the identity-weight trick used
// throughout this test/mlir directory makes the mvm's i32 result equal the
// raw activation values, so any byte lost or reordered in the inserted
// cim.copy would show up directly in the printed numbers.
const char *kStraightLine = R"mlir(
memref.global "private" constant @w : memref<4x4xi8> = dense<[
  [ 1,  0,  0,  0],
  [ 0,  1,  0,  0],
  [ 0,  0,  1,  0],
  [ 0,  0,  0,  1]]>
memref.global "private" constant @a : memref<4xi8> = dense<[5, -3, 100, -128]>
func.func private @cim_print_i32(memref<*xi32>)
func.func @main() {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.get_global @w : memref<4x4xi8>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  %act = memref.get_global @a : memref<4xi8>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
  %u = memref.cast %out : memref<4xi32> to memref<*xi32>
  func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()
  return
}
)mlir";

// A loop-invariant activation feeding an mvm inside an scf.for: exercises
// the hoisting half of the pass, not just insertion. Different activation
// rows per (fake) iteration would defeat the point -- here the SAME
// activation feeds all three iterations on purpose, and each iteration's
// print must match, proving the single hoisted copy is read correctly
// every time, not just on the iteration that happens to run first.
const char *kLoopInvariant = R"mlir(
memref.global "private" constant @w : memref<4x4xi8> = dense<[
  [ 1,  0,  0,  0],
  [ 0,  1,  0,  0],
  [ 0,  0,  1,  0],
  [ 0,  0,  0,  1]]>
memref.global "private" constant @a : memref<4xi8> = dense<[7, -9, 42, -1]>
func.func private @cim_print_i32(memref<*xi32>)
func.func @main() {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.get_global @w : memref<4x4xi8>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  %act = memref.get_global @a : memref<4xi8>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
    %u = memref.cast %out : memref<4xi32> to memref<*xi32>
    func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()
    memref.dealloc %out : memref<4xi32>
  }
  return
}
)mlir";

} // namespace

CIM_TEST(cim_insert_transfers_does_not_change_a_straight_line_result) {
  expectInvariant("straight-line", kStraightLine);
}

CIM_TEST(cim_insert_transfers_does_not_change_a_loop_invariant_result) {
  expectInvariant("loop-invariant", kLoopInvariant);
}
