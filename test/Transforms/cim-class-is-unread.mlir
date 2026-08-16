// The claim in docs/target-format.md's `class:` row -- digital_cim vs
// analog_cim is "the compute-locality axis", parsed and echoed by
// `cim-bench dump-target`, but not read by any compiler pass -- was true
// but, unlike `capabilities.autonomous_control`, unpinned: docs/
// roadmap.md's M4 entry states "grepping every pass confirms none of them
// branch on TargetClass at all", but every existing tiny-*.yaml pair that
// differs in `class:` (tiny-4x4.yaml vs tiny-4x4-4bit.yaml) also differs
// in something else (precision.output_effective_bits), so no test could
// isolate `class:` on its own. A future pass that started branching on
// TargetClass inconsistently could pass every other test in this
// repository and only show up as a silently wrong claim in documentation
// nobody re-checks -- the exact failure mode
// cim-autonomous-control-is-unread.mlir exists to close for
// autonomous_control.
//
// test/targets/tiny-4x4.yaml and test/targets/tiny-4x4-analog.yaml are
// byte-identical except class (digital_cim vs analog_cim) -- see the
// analog fixture's own header. Running the SAME module through the full
// compiler chain against each and diffing the two outputs is the
// strongest available proof that nothing reads the field: not an
// assertion that a specific pass ignores it, but that literally every
// pass's combined output is unaffected, character for character.
//
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-schedule \
// RUN:   --cim-insert-transfers \
// RUN:   --cim-legalize-precision=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   -o %t.digital.mlir
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4-analog.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4-analog.yaml \
// RUN:   --cim-schedule \
// RUN:   --cim-insert-transfers \
// RUN:   --cim-legalize-precision=target-yaml=%S/../targets/tiny-4x4-analog.yaml \
// RUN:   --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4-analog.yaml \
// RUN:   -o %t.analog.mlir
// RUN: diff %t.digital.mlir %t.analog.mlir
// RUN: FileCheck %s < %t.digital.mlir

// A real matmul, run all the way to real cimrt_* calls -- the same shape
// cim-autonomous-control-is-unread.mlir and cim-pipeline-full.mlir's FULL
// run use -- so this exercises every pass class could conceivably matter
// to, not just a hand-picked one.
// CHECK: call @cimrt_mvm
func.func @single_tile_matmul(%act: memref<1x4xi8>, %out: memref<1x4xi32>) {
  %w = memref.get_global @weights_4x4 : memref<4x4xi8>
  linalg.matmul_transpose_b ins(%act, %w : memref<1x4xi8>, memref<4x4xi8>)
                            outs(%out : memref<1x4xi32>)
  return
}
memref.global "private" constant @weights_4x4 : memref<4x4xi8> = dense<1>
