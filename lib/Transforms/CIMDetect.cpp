//===- CIMDetect.cpp - Pass 1: detect CIM candidate matmuls ----*- C++ -*-===//
//
// Spec Sec. 6, Pass 1. Annotates linalg matmuls that are eligible for CIM
// offload; it does NOT convert them. Detection and conversion are separate
// passes so each is independently testable -- a misfiring detector and a
// broken partitioner produce very different bug reports, and keeping them
// apart is what makes that distinction visible.
//
// Eligibility, per the v0.1 contract (spec Sec. 2):
//   - INT8 inputs with a wider integer accumulator
//   - one input operand is a constant, i.e. a weight rather than an
//     activation. A matmul of two activations has nothing to make resident,
//     so weight-stationary hardware buys it nothing.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "cim/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cim-detect"

using namespace mlir;
using namespace mlir::cim;

namespace {

/// Element type of a shaped (tensor or memref) value, or null.
Type shapedElementType(Value value) {
  if (auto shaped = llvm::dyn_cast<ShapedType>(value.getType()))
    return shaped.getElementType();
  return nullptr;
}

/// Is this value a compile-time-known weight rather than a runtime
/// activation? Covers the two shapes a frontend actually produces: an
/// inline constant (tensor semantics, e.g. from torch-mlir) and a global
/// buffer reference (memref semantics, after bufferization).
bool isWeightOperand(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return false; // block argument: an activation supplied by the caller
  if (def->hasTrait<OpTrait::ConstantLike>())
    return true;
  // memref.get_global, without taking a MemRef dialect dependency just for
  // the name check.
  return def->getName().getStringRef() == "memref.get_global";
}

/// The v0.1 contract is INT8 in, wider-integer accumulator out.
bool hasInt8Contract(linalg::LinalgOp op) {
  for (Value input : op.getDpsInputs()) {
    auto elem = llvm::dyn_cast_or_null<IntegerType>(shapedElementType(input));
    if (!elem || elem.getWidth() != 8)
      return false;
  }
  for (Value init : op.getDpsInits()) {
    auto elem = llvm::dyn_cast_or_null<IntegerType>(shapedElementType(init));
    // The accumulator must be wider than the inputs, or partial sums
    // saturate -- the same invariant cim.mvm's verifier enforces.
    if (!elem || elem.getWidth() <= 8)
      return false;
  }
  return !op.getDpsInputs().empty() && !op.getDpsInits().empty();
}

struct CIMDetectPass : public CIMDetectBase<CIMDetectPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    unsigned detected = 0;

    // Declining is legitimate -- most linalg ops in a real module are not
    // CIM candidates and never will be -- so these are remarks, not
    // warnings or errors. What is NOT legitimate is declining in silence,
    // which is what this pass used to do: three bare `return`s, no
    // diagnostic at any level, in a file whose own comment below says
    // convolution "must not be silently accepted". Because cim-partition
    // only walks ops carrying cim.candidate, an undetected op makes the
    // ENTIRE pipeline silent -- a conv, an fp32 matmul, or a matmul whose
    // weight arrives as a function argument compiles to a module with
    // nothing offloaded and no message anywhere explaining why. The very
    // next pass warns four different ways for each candidate it declines
    // ("leaving this candidate unoffloaded"); the front door said nothing.
    //
    // Remarks are off by default and surface under
    // -mlir-print-ir-after-all / a diagnostic handler, so this costs a
    // normal run nothing while making "why did nothing offload?" a
    // question the compiler can answer about itself.
    module.walk([&](linalg::LinalgOp op) {
      // v0.1 handles plain and batched matmul. Convolution is explicitly
      // out of scope (spec Sec. 2) and must not be silently accepted.
      if (!llvm::isa<linalg::MatmulOp, linalg::BatchMatmulOp,
                     linalg::MatmulTransposeBOp>(op.getOperation())) {
        op->emitRemark("cim-detect: not a matmul shape this pass offloads (")
            << op->getName()
            << "); v0.1 accepts linalg.matmul, linalg.batch_matmul and "
               "linalg.matmul_transpose_b only, so this op is left "
               "unoffloaded";
        return;
      }

      if (!hasInt8Contract(op)) {
        op->emitRemark(
            "cim-detect: does not match the INT8-in/wider-accumulator-out "
            "contract (every input i8, every init wider than i8), so this "
            "op is left unoffloaded");
        return;
      }

      // Exactly one input must be a weight. If both are constants the op
      // should have been folded; if neither is, there is nothing to hold
      // resident.
      std::optional<unsigned> weightIndex;
      unsigned constantCount = 0;
      for (auto [index, input] : llvm::enumerate(op.getDpsInputs())) {
        if (isWeightOperand(input)) {
          ++constantCount;
          if (!weightIndex)
            weightIndex = static_cast<unsigned>(index);
        }
      }
      if (constantCount != 1) {
        op->emitRemark("cim-detect: needs exactly one constant (weight) "
                       "operand, found ")
            << constantCount
            << "; weight-stationary hardware has nothing to make resident "
               "otherwise, so this op is left unoffloaded";
        return;
      }

      op->setAttr("cim.candidate", UnitAttr::get(ctx));
      // Which operand to make resident. cim-partition needs this and should
      // not have to re-derive it.
      op->setAttr("cim.weight_operand",
                  IntegerAttr::get(IntegerType::get(ctx, 64), *weightIndex));
      ++detected;
    });

    LLVM_DEBUG(llvm::dbgs()
               << "cim-detect: annotated " << detected << " candidate(s)\n");
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cim::createCIMDetectPass() {
  return std::make_unique<CIMDetectPass>();
}
