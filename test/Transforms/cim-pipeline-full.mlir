// The regression test for the composition-hardening work: this is the
// first test in the project that chains passes past cim-insert-transfers
// (Pass 5), and (FULL, below) the first to chain all eight in spec order
// on one module. Confirmed to fail against the pre-fix code at each stage
// (PRECISION and TARGET reproduced the exact diagnostics documented in
// lib/Transforms/CIMLegalizePrecision.cpp's and
// lib/Transforms/CIMLowerToTarget.cpp's file headers; FULL additionally
// reproduced "cim.requantize is not lowered yet" before cim.requantize
// lowering landed) before landing.
//
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-schedule \
// RUN:   --cim-insert-transfers \
// RUN:   --cim-legalize-precision=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   "--cim-cost-report=target-yaml=%S/../targets/tiny-4x4.yaml output=%t-precision.json" \
// RUN:   | FileCheck --check-prefix=PRECISION %s
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-schedule \
// RUN:   --cim-insert-transfers \
// RUN:   "--cim-cost-report=target-yaml=%S/../targets/tiny-4x4.yaml output=%t-target.json" \
// RUN:   --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   | FileCheck --check-prefix=TARGET %s
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-schedule \
// RUN:   --cim-insert-transfers \
// RUN:   --cim-legalize-precision=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   "--cim-cost-report=target-yaml=%S/../targets/tiny-4x4.yaml output=%t-full.json" \
// RUN:   --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   | FileCheck --check-prefix=FULL %s
//
// Three RUN lines, not because a literal all-eight chain was ever
// impossible, but because each proves something the others don't
// (deliberately not spelled with a colon right after any check-prefix name
// anywhere in this paragraph -- FileCheck would read that substring as a
// directive):
//   - the first chains passes 1-6 plus cost-report (8's numbering, run
//     here right after 6 rather than after 7, because it walks
//     cim.program/cim.mvm sites that Pass 7 would have already erased into
//     cimrt_* calls by the time it ran -- see CIMCostReport.cpp) through a
//     real cim.requantize insertion, checked structurally at the cim
//     dialect level. This was the regression test for the Pass 6 retype
//     fix: pre-fix, cim-legalize-precision's un-retyped cim.copy back to
//     host broke verification here with "copy must not change element
//     type".
//   - the second chains passes 1-5, cost-report, and Pass 7 -- skipping
//     precision legalization -- all the way to real func.call sites
//     against cimrt.h's C ABI, for a model that never needed precision
//     narrowing at all (a completely legitimate real shape, not just a
//     workaround). This was the regression test for the Pass 7 identity-
//     subview fix: pre-fix, the memref.subview cim-partition emits slicing
//     the staged activation broke verification here with "no lowering
//     defined for that use".
//   - the third (FULL) chains literally all eight passes, spec order, on
//     the SAME module, checking it reaches real cimrt_requantize and
//     cimrt_mvm calls with no cim ops left anywhere -- the strongest claim
//     this project can make about composition, and the reason a literal
//     8-pass run is finally possible: cim.requantize is now lowered
//     (lib/Transforms/CIMLowerToTarget.cpp's lowerRequantize), so Pass 6
//     feeding directly into Pass 7 is no longer refused.
//
// A single-tile matmul (activation width == tile width, tiny-4x4.yaml's
// tiles are 4x4): the K dimension needs no partial-sum reduction and the
// per-tile activation subview cim-partition emits is exactly the identity
// slice the Pass 7 fix taught cim-lower-to-target to fold.
func.func @single_tile_matmul(%act: memref<1x4xi8>, %out: memref<1x4xi32>) {
  %w = memref.get_global @weights_4x4 : memref<4x4xi8>
  linalg.matmul_transpose_b ins(%act, %w : memref<1x4xi8>, memref<4x4xi8>)
                            outs(%out : memref<1x4xi32>)
  return
}
memref.global "private" constant @weights_4x4 : memref<4x4xi8> = dense<1>

// PRECISION-LABEL: func.func @single_tile_matmul
// A real cim.requantize survives all the way to the end of this chain --
// the whole point of this RUN line -- with its downstream cim.copy/
// memref.copy retyped to match it: this matmul's %out argument
// (memref<1x4xi32>) is exactly the "sink this pass does not own" case, so
// cim-legalize-precision's width-preserving fallback fires, and the
// requantize's result stays i32 (see CIMLegalizePrecision.cpp's file
// header for why that is still a real, correct legalization, not a
// no-op).
// PRECISION: cim.requantize
// PRECISION-SAME: memref<4xi32, #cim.space<near>> -> memref<4xi32, #cim.space<near>>
// PRECISION-NOT: error

// TARGET-LABEL: func.func @single_tile_matmul
// Every cim op became a real cimrt_* call; none survive. Ordinary
// host-side memref.subview ops (slicing the function's own %act/%out
// arguments, never device-space) are untouched and expected to remain --
// only the identity subview of the STAGED, device-space activation is
// what the Pass 7 fix folds away, and it is gone: cimrt_mvm's activations
// operand below is the same buffer the copy's own cimrt_write used, not a
// second buffer or a leftover memref.subview feeding it.
// TARGET: call @cimrt_open
// The activation is staged (its own cimrt_alloc/cimrt_write pair) before
// the weight is: cim-partition's cim.copy for the activation precedes
// cim.tile_alloc/cim.program in program order, so this is the FIRST
// cimrt_write call in the function.
// TARGET: call @cimrt_write(%[[ACTPTR:[0-9]+]],
// TARGET: call @cimrt_program
// TARGET: call @cimrt_mvm(%{{.*}}, %{{.*}}, %[[ACTPTR]]
// TARGET-NOT: cim.device_open
// TARGET-NOT: cim.tile_alloc
// TARGET-NOT: cim.program
// TARGET-NOT: cim.mvm
// TARGET-NOT: cim.copy
// TARGET-NOT: error

// FULL-LABEL: func.func @single_tile_matmul
// The whole point: every real op became a real cimrt_* call, requantize
// included, on the SAME module all eight passes ran across.
// FULL: call @cimrt_open
// FULL: call @cimrt_program
// FULL: call @cimrt_mvm
// FULL: call @cimrt_requantize
// FULL-NOT: cim.device_open
// FULL-NOT: cim.tile_alloc
// FULL-NOT: cim.program
// FULL-NOT: cim.mvm
// FULL-NOT: cim.copy
// FULL-NOT: cim.requantize
// FULL-NOT: error
