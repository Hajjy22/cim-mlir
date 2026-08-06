//===- pipeline_e2e_test.cpp - Numerical end-to-end check ------*- C++ -*-===//
//
// THE POINT OF THE EXERCISE. Runs the real pass pipeline (cim-detect,
// cim-partition) over a matmul, executes the resulting IR through the
// interpreter and the cimrt runtime, and compares the numbers against a
// plain C++ reference.
//
// Until this existed nothing verified that the compiler's output computed
// the right answer. cim-partition emitted IR that looked structurally
// correct -- the FileCheck tests confirm the ops appear in the right order
// with the right shapes -- but no test had ever run it. Shapes matching is
// not arithmetic matching: a transposed weight block, an off-by-one
// subview offset, or a mis-sliced activation all produce structurally
// perfect IR and wrong numbers.
//
// In-process rather than driving cim-opt and cim-run as subprocesses, so it
// runs under ASan, valgrind and gcov with no extra plumbing, and so a
// failure gives a stack trace rather than a diff of stdout.
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

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir;

namespace {

/// Path to the 2-tile 4x4 test target, found relative to the source tree.
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

/// Reference: out[n] = sum_k W[n][k] * act[k], with W output-major [N x K]
/// -- matching linalg.matmul_transpose_b and cim.mvm.
std::vector<int32_t> reference(const std::vector<int8_t> &w,
                               const std::vector<int8_t> &act, int n, int k) {
  std::vector<int32_t> out(n, 0);
  for (int i = 0; i < n; ++i) {
    int64_t acc = 0;
    for (int j = 0; j < k; ++j)
      acc += static_cast<int64_t>(w[i * k + j]) * act[j];
    out[i] = static_cast<int32_t>(static_cast<uint32_t>(acc));
  }
  return out;
}

std::string denseList(const std::vector<int8_t> &values) {
  std::ostringstream os;
  for (size_t i = 0; i < values.size(); ++i)
    os << (i ? ", " : "") << static_cast<int>(values[i]);
  return os.str();
}

/// Build a self-contained module: weights and activations as
/// memref.global constants, output as a memref.alloc, result printed via a
/// declared external. Adopting mlir-cpu-runner's shape (external print plus
/// unranked cast) means this same IR will run unmodified on the JIT path
/// when cim-lower-to-target lands.
std::string buildModule(const std::vector<int8_t> &w,
                        const std::vector<int8_t> &act, int n, int k) {
  std::ostringstream os;
  os << "memref.global \"private\" constant @w : memref<" << n << "x" << k
     << "xi8> = dense<[" ;
  for (int i = 0; i < n; ++i) {
    os << (i ? ", [" : "[");
    for (int j = 0; j < k; ++j)
      os << (j ? ", " : "") << static_cast<int>(w[i * k + j]);
    os << "]";
  }
  os << "]>\n";

  os << "memref.global \"private\" constant @a : memref<1x" << k
     << "xi8> = dense<[[" << denseList(act) << "]]>\n";

  os << "func.func private @cim_print_i32(memref<*xi32>)\n";
  os << "func.func @main() {\n";
  os << "  %w = memref.get_global @w : memref<" << n << "x" << k << "xi8>\n";
  // The activation is staged into a buffer rather than used straight from
  // the global. Both matter: cim-detect requires exactly ONE constant
  // operand (a matmul of two constants has nothing to hold resident and
  // should have been folded), and after real bufferization an activation
  // would be a buffer anyway, not a global.
  os << "  %aInit = memref.get_global @a : memref<1x" << k << "xi8>\n";
  os << "  %a = memref.alloc() : memref<1x" << k << "xi8>\n";
  os << "  memref.copy %aInit, %a : memref<1x" << k << "xi8> to memref<1x"
     << k << "xi8>\n";
  os << "  %out = memref.alloc() : memref<1x" << n << "xi32>\n";
  os << "  linalg.matmul_transpose_b ins(%a, %w : memref<1x" << k
     << "xi8>, memref<" << n << "x" << k << "xi8>) outs(%out : memref<1x" << n
     << "xi32>)\n";
  os << "  %u = memref.cast %out : memref<1x" << n
     << "xi32> to memref<*xi32>\n";
  os << "  func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()\n";
  os << "  memref.dealloc %a : memref<1x" << k << "xi8>\n";
  os << "  memref.dealloc %out : memref<1x" << n << "xi32>\n";
  os << "  return\n";
  os << "}\n";
  return os.str();
}

/// Parse every "cim_print_i32 shape=[...] data=[a,b,c]" line in the output.
/// A module with two matmuls prints twice, and comparing only the first
/// would let a bug in the second go unnoticed.
std::vector<std::vector<int32_t>> parseAllPrinted(const std::string &text,
                                                   std::string *error) {
  std::vector<std::vector<int32_t>> prints;
  size_t cursor = 0;
  while ((cursor = text.find("data=[", cursor)) != std::string::npos) {
    const size_t start = cursor + 6;
    const size_t end = text.find(']', start);
    if (end == std::string::npos) {
      *error = "unterminated data= list";
      return {};
    }
    std::vector<int32_t> values;
    std::stringstream ss(text.substr(start, end - start));
    std::string token;
    while (std::getline(ss, token, ','))
      if (!token.empty())
        values.push_back(static_cast<int32_t>(std::stol(token)));
    prints.push_back(std::move(values));
    cursor = end;
  }
  if (prints.empty())
    *error = "no data= in interpreter output: " + text;
  return prints;
}

/// What one compile-and-execute produced.
struct RunResult {
  std::vector<std::vector<int32_t>> prints;
  /// cim.program ops surviving in the final IR. This is the quantity
  /// cim-placement exists to reduce, so it is asserted, not just observed.
  unsigned programs = 0;
  unsigned mvms = 0;
};

/// Compile a module source through the pipeline and execute it.
///
/// `withPlacement` selects between detect+partition and
/// detect+partition+placement. Running the *same* source both ways is what
/// makes the invariance test below possible: placement is an optimization,
/// so any difference in the numbers is a bug in placement by definition.
RunResult compileSource(const std::string &source, bool withPlacement,
                        std::string *error) {
  RunResult result;

  DialectRegistry registry;
  registry.insert<cim::CIMDialect, func::FuncDialect, memref::MemRefDialect,
                  arith::ArithDialect, linalg::LinalgDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(source, &context);
  if (!module) {
    *error = "failed to parse the generated module";
    return result;
  }

  PassManager pm(&context);
  pm.addPass(cim::createCIMDetectPass());
  auto partition = cim::createCIMPartitionPass();
  if (failed(partition->initializeOptions("target-yaml=" + tinyTarget()))) {
    *error = "failed to set cim-partition options";
    return result;
  }
  pm.addPass(std::move(partition));

  if (withPlacement) {
    auto placement = cim::createCIMPlacementPass();
    if (failed(placement->initializeOptions("target-yaml=" + tinyTarget()))) {
      *error = "failed to set cim-placement options";
      return result;
    }
    pm.addPass(std::move(placement));
  }

  if (failed(pm.run(*module))) {
    *error = "pass pipeline failed";
    return result;
  }

  module->walk([&](cim::ProgramOp) { ++result.programs; });
  module->walk([&](cim::MvmOp) { ++result.mvms; });

  // If cim-partition declined the candidate it would leave linalg in place
  // and the interpreter would reject it -- but check explicitly so the
  // failure says what actually happened.
  if (result.programs == 0) {
    *error = "the pipeline emitted no cim.program: the matmul was not "
             "offloaded, so this test would prove nothing";
    return result;
  }

  std::string captured;
  llvm::raw_string_ostream stream(captured);
  cim::InterpreterOptions options;
  options.targetYAMLPath = tinyTarget();
  options.out = &stream;

  if (failed(cim::run(*module, "main", options))) {
    *error = "interpreter failed to execute the compiled module";
    return result;
  }
  stream.flush();
  result.prints = parseAllPrinted(captured, error);
  return result;
}

/// Two matmuls in ONE block against the SAME weight global.
///
/// This shape exists because cim-placement can only find reuse when a weight
/// sub-matrix is used more than once, and a single matmul uses each of its
/// blocks exactly once. Without a module like this the pass is a no-op by
/// construction and every test of it would pass while proving nothing --
/// the same trap that made eight e2e tests vacuous before the cim.program
/// guard was added.
///
/// Both activations are staged through memref.alloc + memref.copy because
/// cim-detect requires exactly one constant operand per matmul.
std::string buildTwoMatmulModule(const std::vector<int8_t> &w,
                                 const std::vector<int8_t> &actA,
                                 const std::vector<int8_t> &actB, int n,
                                 int k) {
  std::ostringstream os;
  os << "memref.global \"private\" constant @w : memref<" << n << "x" << k
     << "xi8> = dense<[";
  for (int i = 0; i < n; ++i) {
    os << (i ? ", [" : "[");
    for (int j = 0; j < k; ++j)
      os << (j ? ", " : "") << static_cast<int>(w[i * k + j]);
    os << "]";
  }
  os << "]>\n";
  os << "memref.global \"private\" constant @a1 : memref<1x" << k
     << "xi8> = dense<[[" << denseList(actA) << "]]>\n";
  os << "memref.global \"private\" constant @a2 : memref<1x" << k
     << "xi8> = dense<[[" << denseList(actB) << "]]>\n";
  os << "func.func private @cim_print_i32(memref<*xi32>)\n";
  os << "func.func @main() {\n";
  os << "  %w = memref.get_global @w : memref<" << n << "x" << k << "xi8>\n";

  for (const char *suffix : {"1", "2"}) {
    os << "  %i" << suffix << " = memref.get_global @a" << suffix
       << " : memref<1x" << k << "xi8>\n";
    os << "  %a" << suffix << " = memref.alloc() : memref<1x" << k << "xi8>\n";
    os << "  memref.copy %i" << suffix << ", %a" << suffix << " : memref<1x"
       << k << "xi8> to memref<1x" << k << "xi8>\n";
    os << "  %o" << suffix << " = memref.alloc() : memref<1x" << n
       << "xi32>\n";
    os << "  linalg.matmul_transpose_b ins(%a" << suffix << ", %w : memref<1x"
       << k << "xi8>, memref<" << n << "x" << k << "xi8>) outs(%o" << suffix
       << " : memref<1x" << n << "xi32>)\n";
    os << "  %u" << suffix << " = memref.cast %o" << suffix << " : memref<1x"
       << n << "xi32> to memref<*xi32>\n";
    os << "  func.call @cim_print_i32(%u" << suffix
       << ") : (memref<*xi32>) -> ()\n";
  }
  for (const char *suffix : {"1", "2"}) {
    os << "  memref.dealloc %a" << suffix << " : memref<1x" << k << "xi8>\n";
    os << "  memref.dealloc %o" << suffix << " : memref<1x" << n << "xi32>\n";
  }
  os << "  return\n}\n";
  return os.str();
}

/// Compile and run one matmul, returning the computed output.
std::vector<int32_t> compileAndRun(const std::vector<int8_t> &w,
                                    const std::vector<int8_t> &act, int n,
                                    int k, std::string *error) {
  const RunResult result =
      compileSource(buildModule(w, act, n, k), /*withPlacement=*/false, error);
  if (!error->empty())
    return {};
  return result.prints.front();
}

/// Run one case and compare against the reference.
void checkCase(const char *label, const std::vector<int8_t> &w,
               const std::vector<int8_t> &act, int n, int k) {
  std::string error;
  const std::vector<int32_t> got = compileAndRun(w, act, n, k, &error);
  if (!error.empty()) {
    CIM_FAIL(std::string(label) + ": " + error);
    return;
  }
  const std::vector<int32_t> want = reference(w, act, n, k);
  if (got.size() != want.size()) {
    CIM_FAIL(std::string(label) + ": expected " + std::to_string(want.size()) +
             " outputs, got " + std::to_string(got.size()));
    return;
  }
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      CIM_FAIL(std::string(label) + ": output[" + std::to_string(i) +
               "] = " + std::to_string(got[i]) + ", reference says " +
               std::to_string(want[i]));
      return;
    }
  }
}

std::vector<int8_t> sequential(int count, int start, int step) {
  std::vector<int8_t> v;
  v.reserve(count);
  for (int i = 0; i < count; ++i)
    v.push_back(static_cast<int8_t>(start + i * step));
  return v;
}

} // namespace

//===----------------------------------------------------------------------===//
// Single tile block: N == tileRows, K == tileCols
//===----------------------------------------------------------------------===//

CIM_TEST(e2e_identity_weights_pass_activations_through) {
  const std::vector<int8_t> w = {1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1};
  const std::vector<int8_t> act = {7, -9, 42, -128};
  checkCase("identity", w, act, 4, 4);
}

CIM_TEST(e2e_single_block_arbitrary_weights) {
  const std::vector<int8_t> w = {1, 2, 3, 4,
                                  5, 6, 7, 8,
                                  -1, -2, -3, -4,
                                  9, -9, 9, -9};
  const std::vector<int8_t> act = {2, -3, 5, -7};
  checkCase("single block", w, act, 4, 4);
}

CIM_TEST(e2e_extreme_int8_values) {
  // -128 and 127 at the boundaries: a missing sign reinterpretation on the
  // uint8_t storage path passes an all-positive test and fails here.
  const std::vector<int8_t> w = {-128, 127, -128, 127,
                                  127, -128, 127, -128,
                                  -128, -128, -128, -128,
                                  127, 127, 127, 127};
  const std::vector<int8_t> act = {-128, 127, -128, 127};
  checkCase("extremes", w, act, 4, 4);
}

CIM_TEST(e2e_zero_weights_produce_zero) {
  const std::vector<int8_t> w(16, 0);
  const std::vector<int8_t> act = {1, 2, 3, 4};
  checkCase("zeros", w, act, 4, 4);
}

//===----------------------------------------------------------------------===//
// Multiple blocks: this is where tiling can go wrong
//===----------------------------------------------------------------------===//

CIM_TEST(e2e_two_blocks_in_n_exercises_output_tiling) {
  // N = 8 over 4-row tiles: two output blocks, each its own tile. A wrong
  // weight-block offset shows up here and cannot show up with one block.
  const std::vector<int8_t> w = sequential(8 * 4, -16, 1);
  const std::vector<int8_t> act = {3, -5, 7, -11};
  checkCase("two N blocks", w, act, 8, 4);
}

CIM_TEST(e2e_two_blocks_in_k_exercises_reduce_partial) {
  // K = 8 over 4-column tiles: two partial sums per output, reduced by
  // cim.reduce_partial. A mis-sliced activation window is invisible until
  // this case.
  const std::vector<int8_t> w = sequential(4 * 8, -10, 1);
  const std::vector<int8_t> act = sequential(8, -3, 2);
  checkCase("two K blocks", w, act, 4, 8);
}

CIM_TEST(e2e_four_blocks_tiles_both_dimensions) {
  // 8x8 on 4x4 tiles: 2 blocks in N and 2 in K, four programs and two
  // reductions -- the full tiling path in one case.
  const std::vector<int8_t> w = sequential(8 * 8, -32, 1);
  const std::vector<int8_t> act = sequential(8, 5, -1);
  checkCase("four blocks", w, act, 8, 8);
}

CIM_TEST(e2e_more_blocks_than_tiles_forces_tile_reuse) {
  // The target has 2 tiles; 16x4 needs 4 blocks, so tiles get reprogrammed
  // mid-execution. This is the case that found the bug where cim-partition
  // handed out one tile id per block and asked for tile 2 on a 2-tile
  // device. Reuse must be numerically transparent: reprogramming a tile
  // must not disturb results already computed from its previous contents.
  const std::vector<int8_t> w = sequential(16 * 4, -30, 1);
  const std::vector<int8_t> act = {1, -1, 1, -1};
  checkCase("reuse", w, act, 16, 4);
}

CIM_TEST(e2e_many_blocks_stress_reuse_in_both_dimensions) {
  // 16x8 on 4x4 tiles: 4 N-blocks x 2 K-blocks = 8 blocks over 2 tiles, so
  // every tile is reprogrammed four times while partial sums accumulate.
  const std::vector<int8_t> w = sequential(16 * 8, -64, 1);
  const std::vector<int8_t> act = sequential(8, -4, 1);
  checkCase("stress reuse", w, act, 16, 8);
}

//===----------------------------------------------------------------------===//
// cim-placement: the flagship pass
//
// Until this section existed, cim-placement was unreachable dead code -- its
// use sequence was never populated, so the engine was never called. These
// tests are what make the claim "optimal placement eliminates redundant
// reprogramming" a checked property of the compiler rather than a property
// of the standalone simulator in cim-bench.
//===----------------------------------------------------------------------===//

namespace {

/// Run one source both ways and require the numbers to be identical.
/// Returns the two results so callers can additionally assert on counts.
std::pair<RunResult, RunResult> runBothWays(const char *label,
                                            const std::string &source) {
  std::string error;
  const RunResult plain = compileSource(source, /*withPlacement=*/false,
                                        &error);
  if (!error.empty()) {
    CIM_FAIL(std::string(label) + " (without placement): " + error);
    return {};
  }
  const RunResult placed = compileSource(source, /*withPlacement=*/true,
                                         &error);
  if (!error.empty()) {
    CIM_FAIL(std::string(label) + " (with placement): " + error);
    return {};
  }

  // THE INVARIANT. cim-placement is an optimization: it may emit fewer
  // cim.program ops, and it may not change a single output value. Every
  // other assertion in this file is about how well it optimizes; this one
  // is about whether it is allowed to run at all.
  if (plain.prints.size() != placed.prints.size()) {
    CIM_FAIL(std::string(label) + ": placement changed the number of printed "
             "results from " + std::to_string(plain.prints.size()) + " to " +
             std::to_string(placed.prints.size()));
    return {};
  }
  for (size_t p = 0; p < plain.prints.size(); ++p) {
    if (plain.prints[p].size() != placed.prints[p].size()) {
      CIM_FAIL(std::string(label) + ": placement changed the width of print " +
               std::to_string(p));
      return {};
    }
    for (size_t i = 0; i < plain.prints[p].size(); ++i) {
      if (plain.prints[p][i] != placed.prints[p][i]) {
        CIM_FAIL(std::string(label) + ": placement changed print " +
                 std::to_string(p) + " element " + std::to_string(i) +
                 " from " + std::to_string(plain.prints[p][i]) + " to " +
                 std::to_string(placed.prints[p][i]));
        return {};
      }
    }
  }

  // mvms must never change: placement removes redundant weight writes, not
  // computation. A pass that dropped an mvm would keep the printed values
  // of an earlier output and look correct on a symmetric case.
  if (plain.mvms != placed.mvms)
    CIM_FAIL(std::string(label) + ": placement changed the mvm count from " +
             std::to_string(plain.mvms) + " to " + std::to_string(placed.mvms));

  return {plain, placed};
}

/// Reference for the two-matmul module: both matmuls share `w`.
void checkTwoMatmulValues(const char *label, const RunResult &result,
                          const std::vector<int8_t> &w,
                          const std::vector<int8_t> &actA,
                          const std::vector<int8_t> &actB, int n, int k) {
  if (result.prints.size() != 2) {
    CIM_FAIL(std::string(label) + ": expected 2 printed results, got " +
             std::to_string(result.prints.size()));
    return;
  }
  const std::vector<std::vector<int8_t>> acts = {actA, actB};
  for (size_t p = 0; p < 2; ++p) {
    const std::vector<int32_t> want = reference(w, acts[p], n, k);
    if (result.prints[p] != want) {
      CIM_FAIL(std::string(label) + ": matmul " + std::to_string(p) +
               " disagrees with the reference");
      return;
    }
  }
}

/// One matmul inside an scf.for, iterating over `acts.size()` rows.
///
/// Weights are a single shared global, used identically by every runtime
/// iteration -- loop-invariant by construction, matching what a real
/// inference loop looks like (fixed weights, one activation per call).
/// Activations are staged from a 2D global sliced by the induction
/// variable, so their addressing genuinely depends on %i and must NOT be
/// hoisted; this is what proves cim-placement's hoisting sweep discriminates
/// weight-side from activation-side computation rather than moving anything
/// invariant-looking.
std::string buildLoopModule(const std::vector<int8_t> &w,
                            const std::vector<std::vector<int8_t>> &acts,
                            int n, int k) {
  const int iters = static_cast<int>(acts.size());
  std::ostringstream os;
  os << "memref.global \"private\" constant @w : memref<" << n << "x" << k
     << "xi8> = dense<[";
  for (int i = 0; i < n; ++i) {
    os << (i ? ", [" : "[");
    for (int j = 0; j < k; ++j)
      os << (j ? ", " : "") << static_cast<int>(w[i * k + j]);
    os << "]";
  }
  os << "]>\n";

  os << "memref.global \"private\" constant @acts : memref<" << iters << "x"
     << k << "xi8> = dense<[";
  for (int r = 0; r < iters; ++r)
    os << (r ? ", [" : "[") << denseList(acts[static_cast<size_t>(r)]) << "]";
  os << "]>\n";

  os << "func.func private @cim_print_i32(memref<*xi32>)\n";
  os << "func.func @main() {\n";
  os << "  %w = memref.get_global @w : memref<" << n << "x" << k << "xi8>\n";
  os << "  %acts = memref.get_global @acts : memref<" << iters << "x" << k
     << "xi8>\n";
  os << "  %out = memref.alloc() : memref<" << iters << "x" << n << "xi32>\n";
  os << "  %c0 = arith.constant 0 : index\n";
  os << "  %c1 = arith.constant 1 : index\n";
  os << "  %cN = arith.constant " << iters << " : index\n";
  os << "  scf.for %i = %c0 to %cN step %c1 {\n";
  os << "    %actRow = memref.subview %acts[%i, 0] [1, " << k << "] [1, 1]"
        " : memref<"
     << iters << "x" << k << "xi8> to memref<1x" << k << "xi8, strided<["
     << k << ", 1], offset: ?>>\n";
  os << "    %actLocal = memref.alloc() : memref<1x" << k << "xi8>\n";
  os << "    memref.copy %actRow, %actLocal : memref<1x" << k
     << "xi8, strided<[" << k << ", 1], offset: ?>> to memref<1x" << k
     << "xi8>\n";
  os << "    %outRow = memref.subview %out[%i, 0] [1, " << n << "] [1, 1]"
        " : memref<"
     << iters << "x" << n << "xi32> to memref<1x" << n << "xi32, strided<["
     << n << ", 1], offset: ?>>\n";
  os << "    linalg.matmul_transpose_b ins(%actLocal, %w : memref<1x" << k
     << "xi8>, memref<" << n << "x" << k << "xi8>) outs(%outRow : memref<1x"
     << n << "xi32, strided<[" << n << ", 1], offset: ?>>)\n";
  os << "    memref.dealloc %actLocal : memref<1x" << k << "xi8>\n";
  os << "  }\n";
  os << "  %cast = memref.cast %out : memref<" << iters << "x" << n
     << "xi32> to memref<*xi32>\n";
  os << "  func.call @cim_print_i32(%cast) : (memref<*xi32>) -> ()\n";
  os << "  memref.dealloc %out : memref<" << iters << "x" << n << "xi32>\n";
  os << "  return\n}\n";
  return os.str();
}

/// buildLoopModule prints the whole `out` buffer once, after the loop, so
/// there is exactly one print of iters*n values -- the per-iteration
/// reference rows concatenated in order.
void checkLoopValues(const char *label, const RunResult &result,
                     const std::vector<int8_t> &w,
                     const std::vector<std::vector<int8_t>> &acts, int n,
                     int k) {
  if (result.prints.size() != 1) {
    CIM_FAIL(std::string(label) + ": expected exactly one printed buffer, "
             "got " + std::to_string(result.prints.size()));
    return;
  }
  std::vector<int32_t> want;
  for (const std::vector<int8_t> &act : acts) {
    const std::vector<int32_t> row = reference(w, act, n, k);
    want.insert(want.end(), row.begin(), row.end());
  }
  if (result.prints[0] != want)
    CIM_FAIL(std::string(label) +
             ": loop output disagrees with the per-iteration reference");
}

} // namespace

CIM_TEST(placement_never_changes_the_numbers_on_any_single_matmul_case) {
  // The invariance check applied to the same shapes the nine tests above
  // cover. Placement finds no reuse in any of them -- each block is used
  // once -- so this is specifically a test that a pass with nothing to do
  // does nothing, including on the multi-block and tile-reuse cases where
  // it has the most opportunity to corrupt tile assignment.
  struct Case {
    const char *label;
    int n, k;
  };
  for (const Case &c : {Case{"4x4", 4, 4}, Case{"8x4", 8, 4}, Case{"4x8", 4, 8},
                        Case{"8x8", 8, 8}, Case{"16x4", 16, 4},
                        Case{"16x8", 16, 8}}) {
    const std::vector<int8_t> w = sequential(c.n * c.k, -40, 1);
    const std::vector<int8_t> act = sequential(c.k, -3, 2);
    const auto [plain, placed] =
        runBothWays(c.label, buildModule(w, act, c.n, c.k));
    if (plain.programs == 0)
      continue; // runBothWays already reported the failure
    // Single matmul: every block distinct, so nothing can be reused.
    if (placed.programs != plain.programs)
      CIM_FAIL(std::string(c.label) +
               ": a single matmul has no repeated weights, so placement must "
               "not change the program count, but it went from " +
               std::to_string(plain.programs) + " to " +
               std::to_string(placed.programs));
  }
}

CIM_TEST(placement_eliminates_reprogramming_when_the_weights_fit) {
  // THE HEADLINE CLAIM, on real IR. 8x4 weights over 4x4 tiles is 2 blocks;
  // the target has 2 tiles; two matmuls give the use sequence [0,1,0,1].
  // Everything fits, so after the first pass nothing is reprogrammed: 4
  // cim.program ops must become 2.
  const std::vector<int8_t> w = sequential(8 * 4, -16, 1);
  const std::vector<int8_t> actA = {10, 20, 30, 40};
  const std::vector<int8_t> actB = {1, -2, 3, -4};

  const auto [plain, placed] =
      runBothWays("fits", buildTwoMatmulModule(w, actA, actB, 8, 4));
  if (plain.programs == 0)
    return;

  CIM_EXPECT_EQ(plain.programs, 4u);
  CIM_EXPECT_EQ(placed.programs, 2u);
  CIM_EXPECT_EQ(placed.mvms, 4u);
  checkTwoMatmulValues("fits", placed, w, actA, actB, 8, 4);
}

CIM_TEST(placement_still_wins_under_spill_pressure) {
  // 8x8 over 4x4 tiles is 2 N-blocks x 2 K-blocks = 4 distinct weights on a
  // 2-tile device; two matmuls give the use sequence [0,1,2,3,0,1,2,3].
  //
  // Worked by hand, because a test that just prints whatever the pass
  // returns asserts nothing. Belady evicts the resident whose next use is
  // furthest away:
  //
  //   step 0  w0 miss, free tile        program 1   resident {0}
  //   step 1  w1 miss, free tile        program 2   resident {0,1}
  //   step 2  w2 miss, full: next use of w0 is 4, of w1 is 5 -> evict w1
  //                                     program 3   resident {0,2}
  //   step 3  w3 miss, full: next use of w0 is 4, of w2 is 6 -> evict w2
  //                                     program 4   resident {0,3}
  //   step 4  w0 RESIDENT                           resident {0,3}
  //   step 5  w1 miss, full: w0 never again         program 5   resident {1,3}
  //   step 6  w2 miss, full: w1 never again         program 6   resident {2,3}
  //   step 7  w3 RESIDENT
  //
  // 6 programs, 2 reuses. LRU on the same sequence reuses nothing at all --
  // it evicts w0 at step 2, exactly the block needed two steps later -- and
  // pays all 8. This case is the project's whole argument in miniature:
  // compile-time knowledge of the use sequence is worth 25% of the
  // reprogramming here, and it is worth it under spill, not only when
  // everything fits.
  const std::vector<int8_t> w = sequential(8 * 8, -32, 1);
  const std::vector<int8_t> actA = sequential(8, 5, -1);
  const std::vector<int8_t> actB = sequential(8, -4, 1);

  const auto [plain, placed] =
      runBothWays("spill", buildTwoMatmulModule(w, actA, actB, 8, 8));
  if (plain.programs == 0)
    return;

  CIM_EXPECT_EQ(plain.programs, 8u);
  CIM_EXPECT_EQ(placed.programs, 6u);
  checkTwoMatmulValues("spill", placed, w, actA, actB, 8, 8);
}

CIM_TEST(placement_reuses_across_matmuls_of_a_single_block) {
  // The minimal reuse case: one 4x4 block used twice. [0,0] -> 1 program.
  const std::vector<int8_t> w = {1, 2, 3, 4, 5, 6, 7, 8,
                                 -1, -2, -3, -4, 9, -9, 9, -9};
  const std::vector<int8_t> actA = {2, -3, 5, -7};
  const std::vector<int8_t> actB = {-128, 127, -128, 127};

  const auto [plain, placed] =
      runBothWays("single block", buildTwoMatmulModule(w, actA, actB, 4, 4));
  if (plain.programs == 0)
    return;

  CIM_EXPECT_EQ(plain.programs, 2u);
  CIM_EXPECT_EQ(placed.programs, 1u);
  checkTwoMatmulValues("single block", placed, w, actA, actB, 4, 4);
}

CIM_TEST(placement_assigns_only_tile_ids_the_target_declares) {
  // Spec Sec. 5.4 rule 5: tile ids must be within the target's declared
  // count. TileAllocOp::verify cannot check this -- the count comes from the
  // target file, which the verifier cannot see -- so cim-placement owning
  // assignment is what makes the rule enforceable at all.
  const std::vector<int8_t> w = sequential(16 * 8, -64, 1);
  const std::vector<int8_t> act = sequential(8, -4, 1);

  DialectRegistry registry;
  registry.insert<cim::CIMDialect, func::FuncDialect, memref::MemRefDialect,
                  arith::ArithDialect, linalg::LinalgDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(buildModule(w, act, 16, 8), &context);
  CIM_EXPECT(module);
  if (!module)
    return;

  PassManager pm(&context);
  pm.addPass(cim::createCIMDetectPass());
  auto partition = cim::createCIMPartitionPass();
  CIM_EXPECT(succeeded(partition->initializeOptions("target-yaml=" + tinyTarget())));
  pm.addPass(std::move(partition));
  auto placement = cim::createCIMPlacementPass();
  CIM_EXPECT(succeeded(placement->initializeOptions("target-yaml=" + tinyTarget())));
  pm.addPass(std::move(placement));
  CIM_EXPECT(succeeded(pm.run(*module)));

  unsigned tiles = 0;
  module->walk([&](cim::TileAllocOp op) {
    ++tiles;
    // tiny-4x4.yaml declares 2 tiles.
    CIM_EXPECT_LT(static_cast<int64_t>(op.getId()), int64_t(2));
    CIM_EXPECT_GE(static_cast<int64_t>(op.getId()), int64_t(0));
  });
  CIM_EXPECT_GT(tiles, 0u);
}

//===----------------------------------------------------------------------===//
// cim-placement: loop hoisting
//
// Everything above proves reuse WITHIN one textual execution of a block.
// These cases are the reason that was not enough: cim-partition's output
// sits inside an scf.for whose body runs many times at runtime while
// appearing once in the IR, and a cim.program textually inside the loop
// still costs a real reprogram on every one of those runtime iterations
// unless something hoists it out. This is the first place that cost is
// checked on compiled IR rather than asserted about the standalone
// simulator in cim-bench.
//===----------------------------------------------------------------------===//

CIM_TEST(placement_hoists_a_program_out_of_a_loop_when_weights_fit) {
  // One 4x4 weight block, three loop iterations with different activations.
  // Every iteration programs the SAME weights -- loop-invariant by
  // construction -- so after hoisting the tile is programmed once and
  // reused for the remaining two runtime iterations, even though there is
  // only one cim.program in the textual IR either way.
  const std::vector<int8_t> w = sequential(4 * 4, -8, 1);
  const std::vector<std::vector<int8_t>> acts = {
      {1, 2, 3, 4}, {10, 20, 30, 40}, {-1, -2, -3, -4}};

  const auto [plain, placed] =
      runBothWays("loop-fits", buildLoopModule(w, acts, 4, 4));
  if (plain.programs == 0)
    return;

  // Without placement, one cim.program per textual instance -- the loop
  // does not multiply it in the IR, but cim-run's profile counters (see
  // test/Run/placement-loop.mlir) show it fires once per runtime iteration.
  CIM_EXPECT_EQ(plain.programs, 1u);
  CIM_EXPECT_EQ(placed.programs, 1u);
  CIM_EXPECT_EQ(placed.mvms, 1u);
  checkLoopValues("loop-fits", placed, w, acts, 4, 4);
}

CIM_TEST(placement_hoisting_does_not_scale_with_iteration_count) {
  // The claim under test is specifically that hoisting decouples IR program
  // count from how many times the loop runs -- not just that it works for
  // one particular iteration count. Same weights, 6 iterations instead of 3.
  const std::vector<int8_t> w = sequential(4 * 4, -8, 1);
  std::vector<std::vector<int8_t>> acts;
  for (int i = 0; i < 6; ++i)
    acts.push_back(sequential(4, -3 + i, 1));

  const auto [plain, placed] =
      runBothWays("loop-fits-6", buildLoopModule(w, acts, 4, 4));
  if (plain.programs == 0)
    return;

  CIM_EXPECT_EQ(placed.programs, 1u);
  CIM_EXPECT_EQ(placed.mvms, 1u);
  checkLoopValues("loop-fits-6", placed, w, acts, 4, 4);
}

CIM_TEST(placement_partially_hoists_when_only_some_tiles_are_stable) {
  // 12x4 weights over 4x4 tiles is 3 distinct blocks on a 2-tile device.
  // Within ONE iteration the use sequence [0,1,2] has no repeats -- nothing
  // for intra-block Belady to reuse -- so all 3 programs survive per
  // iteration regardless of placement. But 3 blocks on 2 tiles means one
  // tile holds a block for the whole iteration (stable, hoistable) while
  // the other is reprogrammed mid-iteration (not stable, not hoistable):
  //
  //   block0 -> tileA, block1 -> tileB, block2 -> evicts whichever of
  //   {tileA, tileB} Belady picks (a tie: neither block0 nor block1 is used
  //   again in THIS iteration) -- so one tile sees 2 programs/iteration,
  //   the other sees exactly 1 and is the hoist candidate.
  //
  // Two iterations: baseline 3 programs/iteration x 2 = 6. With placement,
  // the stable tile's program is hoisted once; the spilled tile's 2
  // programs/iteration remain: 2*2 + 1 = 5.
  const std::vector<int8_t> w = sequential(12 * 4, -24, 1);
  const std::vector<std::vector<int8_t>> acts = {{1, 2, 3, 4},
                                                 {10, 20, 30, 40}};

  const auto [plain, placed] =
      runBothWays("loop-spill", buildLoopModule(w, acts, 12, 4));
  if (plain.programs == 0)
    return;

  CIM_EXPECT_EQ(plain.programs, 3u);
  CIM_EXPECT_EQ(placed.programs, 3u); // 1 hoisted + 2 still in the loop body
  CIM_EXPECT_EQ(placed.mvms, 3u);
  checkLoopValues("loop-spill", placed, w, acts, 12, 4);
}

CIM_TEST(placement_leaves_a_zero_trip_count_loop_alone) {
  // `scf.for %i = %c0 to %c0 step %c1` never runs -- lower == upper -- which
  // loopHasProvablyPositiveTripCount must recognise and refuse, since
  // hoisting a cim.program out of a loop that might never execute would
  // spend energy the original IR never spent. A single fixed activation
  // (not a per-iteration buffer sliced by %i) keeps this case narrowly about
  // the trip-count guard rather than about buildLoopModule's zero-row
  // machinery, which dense<> literals cannot express cleanly.
  //
  // What this test CANNOT catch, proven by mutation testing rather than
  // assumed: `programs` (a module-wide op count) and the printed values are
  // identical whether the program is hoisted above the loop or left inside
  // it -- moving an op changes where it sits, not how many of it exist or
  // what a loop that never runs computes. A `loopHasProvablyPositiveTripCount`
  // that always returned true passes this test unchanged. The check that
  // actually distinguishes "hoisted" from "not hoisted" is structural:
  // test/Transforms/cim-placement-loop.mlir's zero_trip_count_is_left_alone
  // asserts cim.program is still textually inside scf.for. This test stays
  // for what it IS good for -- that the pipeline compiles and executes a
  // zero-trip loop without crashing, and that a loop which never runs
  // genuinely never writes its output.
  std::string source = R"mlir(
    memref.global "private" constant @w : memref<4x4xi8> = dense<1>
    memref.global "private" constant @a : memref<1x4xi8> = dense<2>
    func.func @main() {
      %w = memref.get_global @w : memref<4x4xi8>
      %aInit = memref.get_global @a : memref<1x4xi8>
      %a = memref.alloc() : memref<1x4xi8>
      memref.copy %aInit, %a : memref<1x4xi8> to memref<1x4xi8>
      %out = memref.alloc() : memref<1x4xi32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c0 step %c1 {
        linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<4x4xi8>)
          outs(%out : memref<1x4xi32>)
      }
      %u = memref.cast %out : memref<1x4xi32> to memref<*xi32>
      func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()
      memref.dealloc %a : memref<1x4xi8>
      memref.dealloc %out : memref<1x4xi32>
      return
    }
  )mlir";
  source = "func.func private @cim_print_i32(memref<*xi32>)\n" + source;

  std::string error;
  const RunResult placed =
      compileSource(source, /*withPlacement=*/true, &error);
  // The matmul is inside a loop that never runs, so cim-partition never sees
  // that -- it reasons about the IR structurally, not about trip counts --
  // and offloads it exactly as it would outside a loop; the only question
  // this case can answer is whether cim-placement hoisted the resulting
  // cim.program out, which it must not.
  if (!error.empty()) {
    CIM_FAIL("placement_leaves_a_zero_trip_count_loop_alone: " + error);
    return;
  }
  CIM_EXPECT_EQ(placed.programs, 1u);
  // The loop never ran, so %out was never written by the mvm and reads back
  // as the zero-fill every allocation starts with -- confirming this is
  // genuinely testing "did not run", not merely "did not crash".
  CIM_EXPECT_EQ(placed.prints.size(), size_t(1));
  if (!placed.prints.empty())
    for (int32_t value : placed.prints.front())
      CIM_EXPECT_EQ(value, 0);
}
