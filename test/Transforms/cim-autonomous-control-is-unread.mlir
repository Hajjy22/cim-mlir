// The claim in docs/target-format.md's `capabilities:` table --
// "autonomous_control: ...not read by any pass -- v0.1's execution model
// has nothing host-less to drive" -- was true but unpinned: every existing
// tiny-*.yaml fixture declares autonomous_control: false, so nothing had
// ever run the full pipeline with it true. A future pass that started
// reading this flag inconsistently (say, cim-schedule skipping a barrier
// it thought a host-less target wouldn't need) could pass every other
// test in this repository and only show up as a silently wrong claim in
// documentation nobody re-checks.
//
// test/targets/tiny-4x4.yaml and test/targets/tiny-4x4-autonomous.yaml are
// byte-identical except capabilities.autonomous_control (false vs true) --
// see the autonomous fixture's own header. Running the SAME module through
// the full compiler chain against each and diffing the two outputs is the
// strongest available proof that nothing reads the flag: not an assertion
// that a specific pass ignores it, but that literally every pass's
// combined output is unaffected, character for character.
//
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-schedule \
// RUN:   --cim-insert-transfers \
// RUN:   --cim-legalize-precision=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   -o %t.control-false.mlir
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4-autonomous.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4-autonomous.yaml \
// RUN:   --cim-schedule \
// RUN:   --cim-insert-transfers \
// RUN:   --cim-legalize-precision=target-yaml=%S/../targets/tiny-4x4-autonomous.yaml \
// RUN:   --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4-autonomous.yaml \
// RUN:   -o %t.control-true.mlir
// RUN: diff %t.control-false.mlir %t.control-true.mlir
// RUN: FileCheck %s < %t.control-false.mlir

// A real matmul, run all the way to real cimrt_* calls -- the same shape
// cim-pipeline-full.mlir's FULL run uses -- so this exercises every pass
// autonomous_control could conceivably matter to (cim-schedule's barrier
// placement most plausibly, since spec Sec. 3.1 files the control-model
// axis there), not just a hand-picked one.
// CHECK: call @cimrt_mvm
func.func @single_tile_matmul(%act: memref<1x4xi8>, %out: memref<1x4xi32>) {
  %w = memref.get_global @weights_4x4 : memref<4x4xi8>
  linalg.matmul_transpose_b ins(%act, %w : memref<1x4xi8>, memref<4x4xi8>)
                            outs(%out : memref<1x4xi32>)
  return
}
memref.global "private" constant @weights_4x4 : memref<4x4xi8> = dense<1>
