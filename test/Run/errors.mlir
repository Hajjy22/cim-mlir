// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=nope %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-ENTRY %s
// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=takes_args %s 2>&1 \
// RUN:   | FileCheck --check-prefix=ARGS %s
// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=declared_only %s 2>&1 \
// RUN:   | FileCheck --check-prefix=EXTERNAL %s
// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=unmodelled %s 2>&1 \
// RUN:   | FileCheck --check-prefix=UNMODELLED %s
// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=iter_args %s 2>&1 \
// RUN:   | FileCheck --check-prefix=ITER-ARGS %s
// RUN: not cim-run --target-yaml=%S/no-such-target.yaml --entry=opens_a_device %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BAD-TARGET %s
// RUN: not cim-run %s 2>&1 | FileCheck --check-prefix=NO-TARGET %s
// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=sub_byte_alloc %s 2>&1 \
// RUN:   | FileCheck --check-prefix=SUB-BYTE %s

// The interpreter's refusals are load-bearing. Its whole justification is
// that it never produces a plausible number it cannot stand behind, so every
// "I do not model this" path has to actually fire and actually say why. A
// silent skip here would look exactly like a correct run.

func.func @takes_args(%arg0: memref<4xi8>) {
  return
}

func.func private @declared_only()

func.func @unmodelled() {
  // arith.constant and scf.for are modelled now (loop hoisting needs both to
  // be checkable numerically); arith.addi is not, and cim-partition never
  // emits it either, so it stays a clean example of "valid IR the
  // interpreter has no model for."
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %sum = arith.addi %c1, %c2 : i32
  return
}

func.func @iter_args() {
  // cim-partition never emits a loop with loop-carried values -- every cim
  // op sequence it produces communicates through side effects, not through
  // scf.for results -- so the interpreter refuses rather than attempting to
  // merge a value across iterations it was never designed to track.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %init = arith.constant 0 : i32
  %r = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %init) -> i32 {
    scf.yield %acc : i32
  }
  return
}

// Not tested here: "scf.for bounds must be values the interpreter can
// evaluate" (runScfFor) and "memref.subview offset ... is not a
// compile-time constant or a value the interpreter has bound" (runSubView,
// the dynamic-offset case that is not a loop induction variable). Both
// guards are real and stay as defence in depth, but neither has a reachable
// input: the only way to produce an index-typed SSA value the interpreter
// does NOT bind is a function argument, and cim-run requires a zero-argument
// entry (see @takes_args above) -- so reaching either guard through
// cim-run's CLI would first trip that check instead. Every op capable of
// producing an index either resolves (arith.constant, a bound induction
// variable) or is itself unmodelled and errors before its result could be
// used (@unmodelled). They should get a test in the same change that adds
// an index-producing op the interpreter models but does not track, e.g. if
// arith.addi on index values is ever needed.

func.func @opens_a_device() {
  %d = cim.device_open {target = "tiny-4x4"} : !cim.device<"tiny-4x4">
  return
}

// i4 is an accepted weight_dtype in the target schema (test/python/schema.py
// generates it), so a sub-byte element type reaching the interpreter is
// in-domain, not hypothetical. bitWidth / 8 truncates to 0 for i4 -- a
// zero-byte allocation with no diagnostic -- unless runAlloc refuses it
// first.
func.func @sub_byte_alloc() {
  %m = memref.alloc() : memref<4xi4>
  return
}

// NO-ENTRY: no such entry function: nope
// ARGS: zero-argument entry functions
// EXTERNAL: entry function has no body
// UNMODELLED: operation not supported by the cim interpreter: arith.addi
// ITER-ARGS: loop-carried values
// BAD-TARGET: cimrt_open
// NO-TARGET: --target-yaml is required
// SUB-BYTE: narrower than a byte
