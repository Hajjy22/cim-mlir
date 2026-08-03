// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=nope %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-ENTRY %s
// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=takes_args %s 2>&1 \
// RUN:   | FileCheck --check-prefix=ARGS %s
// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=declared_only %s 2>&1 \
// RUN:   | FileCheck --check-prefix=EXTERNAL %s
// RUN: not cim-run --target-yaml=%S/../targets/tiny-4x4.yaml --entry=unmodelled %s 2>&1 \
// RUN:   | FileCheck --check-prefix=UNMODELLED %s
// RUN: not cim-run --target-yaml=%S/no-such-target.yaml --entry=opens_a_device %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BAD-TARGET %s
// RUN: not cim-run %s 2>&1 | FileCheck --check-prefix=NO-TARGET %s

// The interpreter's refusals are load-bearing. Its whole justification is
// that it never produces a plausible number it cannot stand behind, so every
// "I do not model this" path has to actually fire and actually say why. A
// silent skip here would look exactly like a correct run.

func.func @takes_args(%arg0: memref<4xi8>) {
  return
}

func.func private @declared_only()

func.func @unmodelled() {
  // arith.constant is perfectly valid IR that cim-partition never emits, so
  // the interpreter has no model for it. It must say so rather than step
  // over it.
  %c = arith.constant 7 : i32
  return
}

// Not tested here: the interpreter's "memref.subview with a dynamic offset"
// guard. It has no test because it currently has no reachable input. A
// dynamic offset must come from an index-producing op, and every such op --
// arith.constant included, as @unmodelled above shows -- is refused by the
// .Default branch before the subview is ever reached. The guard stays as
// defence in depth for when index-producing ops are modelled; it should get
// a test in the same change that makes it reachable.

func.func @opens_a_device() {
  %d = cim.device_open {target = "tiny-4x4"} : !cim.device<"tiny-4x4">
  return
}

// NO-ENTRY: no such entry function: nope
// ARGS: zero-argument entry functions
// EXTERNAL: entry function has no body
// UNMODELLED: operation not supported by the cim interpreter: arith.constant
// BAD-TARGET: cimrt_open
// NO-TARGET: --target-yaml is required
