// Pass-level failures. Uses `not cim-opt ... | FileCheck` rather than
// --verify-diagnostics because these diagnostics are attached to the module
// with an unknown location, which expected-error handles awkwardly.

// RUN: not cim-opt %s --cim-partition 2>&1 | FileCheck --check-prefix=NO-TARGET %s
// RUN: not cim-opt %s --cim-partition=target-yaml=/nonexistent/target.yaml 2>&1 \
// RUN:   | FileCheck --check-prefix=BAD-PATH %s
// RUN: not cim-opt %s --cim-partition=target-yaml=%S/malformed-target.yaml 2>&1 \
// RUN:   | FileCheck --check-prefix=MALFORMED %s

// The tile geometry has to come from somewhere; guessing it would silently
// produce a partition for hardware that does not exist.
// NO-TARGET: requires -target-yaml

// BAD-PATH: failed to open target file

// MALFORMED: missing required field
func.func @noop() {
  return
}
