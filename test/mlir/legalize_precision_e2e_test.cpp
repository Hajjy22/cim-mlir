//===- legalize_precision_e2e_test.cpp - cim.requantize arithmetic ------===//
//
// Unlike cim-schedule's cim.barrier, cim.requantize genuinely changes
// values -- there is no "does not change the answer" invariant to check
// here. So this file hand-writes cim IR (device_open/tile_alloc/program/mvm)
// rather than running the full cim-detect/cim-partition/cim-placement/
// cim-legalize-precision pipeline: that pipeline hits the documented
// cim-partition <-> cim-legalize-precision integration gap (see the file
// header comment in lib/Transforms/CIMLegalizePrecision.cpp), and this
// file's job is checking the INTERPRETER's arithmetic against cim.requantize
// as a dialect op, independent of which pass ends up emitting it.
//
// Expected quantized values are computed independently in C++
// (expectedRequantize below) and checked against what the interpreter
// actually printed -- a real differential, in the same spirit as the numpy
// and PyYAML ones, not just "did it run".
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Dialect/CIMDialect.h"
#include "cim/Dialect/CIMOps.h"
#include "cim/Interpreter/Interpreter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir;

namespace {

/// Independent reference implementation of cim.requantize's arithmetic,
/// matching the formula documented in Interpreter.cpp's runRequantize:
/// round-half-away-from-zero, then clamp to the signed range effective_bits
/// can hold. Computed here from first principles (not by calling into the
/// interpreter's own code) so a bug shared between the two would not be
/// invisible to this test.
int64_t expectedRequantize(int32_t value, double scale, int32_t zeroPoint,
                            uint32_t effectiveBits) {
  const double scaled = static_cast<double>(value) / scale;
  int64_t quantized =
      zeroPoint + static_cast<int64_t>(std::llround(scaled));
  const int64_t clampMin = -(static_cast<int64_t>(1) << (effectiveBits - 1));
  const int64_t clampMax = (static_cast<int64_t>(1) << (effectiveBits - 1)) - 1;
  if (quantized < clampMin)
    quantized = clampMin;
  if (quantized > clampMax)
    quantized = clampMax;
  return quantized;
}

/// A single cim.mvm against an identity weight matrix, so the i32
/// accumulator equals `acts` exactly -- what cim.requantize sees as input
/// is then under this test's direct control, not derived from a real dot
/// product it would also have to get right. Weights and activations are
/// memref.global dense<> constants (the interpreter's only supported source
/// of constant memref data -- it does not model memref.store/memref.load),
/// matching the pattern every other e2e test in this directory uses.
std::string buildModule(llvm::ArrayRef<int32_t> acts, double scale,
                         int32_t zeroPoint, uint32_t effectiveBits) {
  const size_t n = acts.size();
  std::ostringstream os;
  os << "memref.global \"private\" constant @w : memref<" << n << "x" << n
     << "xi8> = dense<[";
  for (size_t row = 0; row < n; ++row) {
    os << (row ? ",[" : "[");
    for (size_t col = 0; col < n; ++col)
      os << (col ? "," : "") << (row == col ? 1 : 0);
    os << "]";
  }
  os << "]>\n";
  os << "memref.global \"private\" constant @a : memref<" << n << "xi8> = dense<[";
  for (size_t i = 0; i < n; ++i)
    os << (i ? "," : "") << acts[i];
  os << "]>\n";
  os << "func.func private @cim_print_i8(memref<*xi8>)\n";
  os << "func.func @main() {\n";
  os << "  %dev = cim.device_open {target = \"t\"} : !cim.device<\"t\">\n";
  os << "  %t = cim.tile_alloc %dev {id = 0 : i64} "
        ": (!cim.device<\"t\">) -> !cim.tile<"
     << n << "x" << n << "xi8>\n";
  os << "  %w = memref.get_global @w : memref<" << n << "x" << n << "xi8>\n";
  os << "  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64} "
        ": (!cim.tile<"
     << n << "x" << n << "xi8>, memref<" << n << "x" << n
     << "xi8>) -> !cim.resident<" << n << "x" << n << "xi8>\n";
  os << "  %act = memref.get_global @a : memref<" << n << "xi8>\n";
  os << "  %out = cim.mvm %r, %act : (!cim.resident<" << n << "x" << n
     << "xi8>, memref<" << n << "xi8>) -> memref<" << n << "xi32>\n";
  // %.6f, not operator<<, so an integral scale (e.g. 1.0) still parses as a
  // float literal rather than "1", which the MLIR parser rejects for an f32
  // attribute ("unexpected decimal integer literal for a floating point
  // value").
  char scaleBuf[64];
  std::snprintf(scaleBuf, sizeof(scaleBuf), "%.6f", scale);
  os << "  %rq = cim.requantize %out {scale = " << scaleBuf
     << " : f32, zero_point = " << zeroPoint
     << " : i32, effective_bits = " << effectiveBits << " : i32} : memref<"
     << n << "xi32> -> memref<" << n << "xi8>\n";
  os << "  %u = memref.cast %rq : memref<" << n << "xi8> to memref<*xi8>\n";
  os << "  func.call @cim_print_i8(%u) : (memref<*xi8>) -> ()\n";
  os << "  return\n";
  os << "}\n";
  return os.str();
}

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

/// Parse the sole "data=[a,b,c]" the module above ever prints.
std::vector<int64_t> parsePrinted(const std::string &text, std::string *error) {
  const size_t start = text.find("data=[");
  if (start == std::string::npos) {
    *error = "no data= in interpreter output: " + text;
    return {};
  }
  const size_t begin = start + 6;
  const size_t end = text.find(']', begin);
  if (end == std::string::npos) {
    *error = "unterminated data= list";
    return {};
  }
  std::vector<int64_t> values;
  std::stringstream ss(text.substr(begin, end - begin));
  std::string token;
  while (std::getline(ss, token, ','))
    if (!token.empty())
      values.push_back(std::stol(token));
  return values;
}

void expectRequantized(const char *label, llvm::ArrayRef<int32_t> acts,
                       double scale, int32_t zeroPoint,
                       uint32_t effectiveBits) {
  DialectRegistry registry;
  registry.insert<cim::CIMDialect, func::FuncDialect, memref::MemRefDialect,
                  arith::ArithDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  const std::string source =
      buildModule(acts, scale, zeroPoint, effectiveBits);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  if (!module) {
    CIM_FAIL(std::string(label) + ": failed to parse the generated module");
    return;
  }

  std::string captured;
  llvm::raw_string_ostream stream(captured);
  cim::InterpreterOptions options;
  options.targetYAMLPath = tinyTarget();
  options.out = &stream;
  if (failed(cim::run(*module, "main", options))) {
    CIM_FAIL(std::string(label) +
             ": interpreter failed to execute the compiled module");
    return;
  }
  stream.flush();

  std::string error;
  std::vector<int64_t> printed = parsePrinted(captured, &error);
  if (!error.empty()) {
    CIM_FAIL(std::string(label) + ": " + error);
    return;
  }
  CIM_EXPECT_EQ(printed.size(), acts.size());
  for (size_t i = 0; i < acts.size() && i < printed.size(); ++i) {
    const int64_t expected =
        expectedRequantize(acts[i], scale, zeroPoint, effectiveBits);
    CIM_EXPECT_EQ(printed[i], expected);
  }
}

} // namespace

// scale=2.0, zero_point=3, effective_bits=8 (no clamping reachable at this
// width from these inputs): exercises the general op semantics -- a
// fractional scale and a nonzero zero_point, not just the identity
// parameters cim-legalize-precision itself always emits (scale=1.0,
// zero_point=0) -- plus round-half-away-from-zero on both a tie (7/2=3.5)
// and a value that already divides evenly.
CIM_TEST(cim_requantize_applies_scale_zero_point_and_rounds_half_away_from_zero) {
  const int32_t acts[] = {100, -50, 7, -128};
  expectRequantized("scale/zero_point/rounding", acts, /*scale=*/2.0,
                    /*zeroPoint=*/3, /*effectiveBits=*/8);
}

// scale=1.0, zero_point=0, effective_bits=4 -- exactly what
// cim-legalize-precision emits on a sub-8-bit target (spec Sec. 6) -- with
// inputs chosen to hit BOTH the upper clamp (127 -> 7) and the lower clamp
// (-128 -> -8), plus one value already inside [-8, 7] that must pass
// through unchanged. Clamping is this op's entire reason to exist on an
// analog target (spec Sec. 5.3): a test that never reaches the clamp would
// not actually be testing it.
CIM_TEST(cim_requantize_clamps_to_effective_bits) {
  const int32_t acts[] = {127, -128, 5, -8};
  expectRequantized("effective_bits clamp", acts, /*scale=*/1.0,
                    /*zeroPoint=*/0, /*effectiveBits=*/4);
}

// A negative zero_point together with a fractional scale on the OTHER
// rounding tie (-7/2 = -3.5 -> -4, away from zero in the negative
// direction) -- the case most likely to break if the rounding used
// std::round's or std::trunc's asymmetric behavior instead of a genuinely
// symmetric round-half-away-from-zero.
CIM_TEST(cim_requantize_rounds_negative_ties_away_from_zero_too) {
  const int32_t acts[] = {-7, 7, -9, 9};
  expectRequantized("negative rounding ties", acts, /*scale=*/2.0,
                    /*zeroPoint=*/-1, /*effectiveBits=*/8);
}
