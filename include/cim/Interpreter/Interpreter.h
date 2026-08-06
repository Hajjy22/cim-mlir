//===- Interpreter.h - Execute cim IR against cimrt ------------*- C++ -*-===//
//
// Executes the IR that cim-partition emits, so its numerical results can be
// checked against a reference. Until this existed, nothing anywhere verified
// that the compiler's output computed the right answer -- cim-partition
// produced plausible-looking IR whose semantics had never been run.
//
// Why an interpreter rather than lowering to LLVM and JIT-ing: both execute
// the same IR and prove the same property, but a JIT conflates two failure
// modes. When the numbers come out wrong you cannot tell whether
// cim-partition mis-tiled the weights or the memref->LLVM descriptor
// lowering is off, and isolating cim-partition's semantics is the entire
// point. This also pins down the dialect's execution semantics, which
// nothing previously stated -- e.g. that cim.mvm's result is a fresh
// allocation rather than an in-place write.
//
// Memory model: an allocation whose memref type carries
// #cim.space<near|insitu> is backed by a real cimrt_buffer; host-space
// allocations are plain host memory. So every byte the IR claims crosses a
// space boundary really does cross the cimrt boundary, and the runtime is
// exercised by construction rather than mocked.
//
// Control flow: scf.for is executed with a genuinely bound induction
// variable, which is what makes cim-placement's loop hoisting checkable
// numerically rather than only structurally. Scope is deliberately narrow --
// bounds must be arith.constant (a dynamic trip count would have to be
// guessed, which this interpreter never does), and the loop must carry no
// iter_args (cim-partition's output never needs one; every cim op sequence
// communicates through side effects, not through scf.for results). A
// memref.subview's offset may now be a bound index value -- typically the
// induction variable -- rather than only a compile-time constant; anything
// else is still the same hard error it always was.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_INTERPRETER_INTERPRETER_H
#define CIM_INTERPRETER_INTERPRETER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace cim {

struct InterpreterOptions {
  /// Target description passed to cimrt_open. Overrides the target name on
  /// cim.device_open, which is a bare name rather than a path.
  std::string targetYAMLPath;

  /// Where @cim_print_* calls write. Defaults to llvm::outs().
  llvm::raw_ostream *out = nullptr;

  /// Print the cimrt profile counters after execution.
  bool dumpProfile = false;
};

/// Execute `entry` -- a zero-argument func.func in `module` -- to
/// completion.
///
/// Any operation the interpreter does not model is a hard error with a
/// source location, never a silent skip: skipping one op would produce a
/// wrong-but-plausible number, which is the exact failure this whole
/// mechanism exists to detect.
LogicalResult run(ModuleOp module, llvm::StringRef entry,
                  const InterpreterOptions &options);

} // namespace cim
} // namespace mlir

#endif // CIM_INTERPRETER_INTERPRETER_H
