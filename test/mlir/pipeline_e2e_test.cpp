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

/// Parse a printed "cim_print_i32 shape=[...] data=[a,b,c]" line.
std::vector<int32_t> parsePrinted(const std::string &text, std::string *error) {
  const size_t dataPos = text.find("data=[");
  if (dataPos == std::string::npos) {
    *error = "no data= in interpreter output: " + text;
    return {};
  }
  const size_t start = dataPos + 6;
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
  return values;
}

/// Compile and run one matmul, returning the computed output.
std::vector<int32_t> compileAndRun(const std::vector<int8_t> &w,
                                    const std::vector<int8_t> &act, int n,
                                    int k, std::string *error) {
  DialectRegistry registry;
  registry.insert<cim::CIMDialect, func::FuncDialect, memref::MemRefDialect,
                  arith::ArithDialect, linalg::LinalgDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  const std::string source = buildModule(w, act, n, k);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(source, &context);
  if (!module) {
    *error = "failed to parse the generated module";
    return {};
  }

  PassManager pm(&context);
  pm.addPass(cim::createCIMDetectPass());
  auto partition = cim::createCIMPartitionPass();
  if (failed(partition->initializeOptions("target-yaml=" + tinyTarget()))) {
    *error = "failed to set cim-partition options";
    return {};
  }
  pm.addPass(std::move(partition));

  if (failed(pm.run(*module))) {
    *error = "pass pipeline failed";
    return {};
  }

  // If cim-partition declined the candidate it would leave linalg in place
  // and the interpreter would reject it -- but check explicitly so the
  // failure says what actually happened.
  bool sawProgram = false;
  module->walk([&](cim::ProgramOp) { sawProgram = true; });
  if (!sawProgram) {
    *error = "cim-partition emitted no cim.program: the matmul was not "
             "offloaded, so this test would prove nothing";
    return {};
  }

  std::string captured;
  llvm::raw_string_ostream stream(captured);
  cim::InterpreterOptions options;
  options.targetYAMLPath = tinyTarget();
  options.out = &stream;

  if (failed(cim::run(*module, "main", options))) {
    *error = "interpreter failed to execute the compiled module";
    return {};
  }
  stream.flush();
  return parsePrinted(captured, error);
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
