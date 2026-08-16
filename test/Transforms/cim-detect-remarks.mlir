// RUN: cim-opt %s --cim-detect --split-input-file \
// RUN:   --mlir-print-debuginfo=false 2>&1 >/dev/null | FileCheck %s
// Second pass over the same output, with its own prefix, purely to bound
// the TOTAL remark count. It needs to be a separate invocation: the
// per-message checks below consume the three remarks as they match them,
// leaving nothing for a count directive placed after them to find.
// RUN: cim-opt %s --cim-detect --split-input-file \
// RUN:   --mlir-print-debuginfo=false 2>&1 >/dev/null \
// RUN:   | FileCheck --check-prefix=TOTAL %s

// WHY THIS FILE EXISTS
// ====================
// cim-detect.mlir already checks that these ops do NOT get a cim.candidate
// attribute. That is only half the contract. Because cim-partition walks
// ONLY ops carrying cim.candidate, an op this pass declines is invisible to
// every later pass -- so a declined op means the whole pipeline emits a
// module with nothing offloaded. Before this, it did that in total silence:
// three bare `return`s with no diagnostic at any level, in a file whose own
// comment says convolution "must not be silently accepted".
//
// The contrast was with the very next pass: cim-partition warns four
// different ways for each candidate it declines, each ending "leaving this
// candidate unoffloaded". The front door of the pipeline said nothing at
// all, so "why did nothing offload?" had no answer the compiler could give
// about itself.
//
// These are remarks rather than warnings on purpose: declining is
// legitimate and common (most linalg ops in a real module are not CIM
// candidates), so a normal run should stay quiet. What matters is that the
// reason EXISTS and is retrievable, which is what this file pins.

// A NOTE ON HOW THIS IS CHECKED. --split-input-file concatenates every
// section's diagnostics into one stream, so a bare `CHECK-NOT: remark` at
// the end would match remarks emitted by EARLIER sections and fail for a
// reason unrelated to the case it is guarding. Every decline message also
// ends in the same "left unoffloaded" phrase, so CHECK-SAME on that phrase
// is ambiguous across all three. Both problems are solved the same way:
// match each message on the one phrase unique to it, in order, and then
// pin the TOTAL number of remarks. Exactly three, over four sections, is
// what simultaneously proves each decline is reported AND that the valid
// candidate in the fourth section is not.

// A convolution: right element types, wrong op shape entirely.
func.func @conv_reports_the_shape_reason(%in: tensor<1x8x8x4xi8>,
                                          %init: tensor<1x6x6x4xi32>)
    -> tensor<1x6x6x4xi32> {
  %filter = arith.constant dense<1> : tensor<3x3x4x4xi8>
  // CHECK: remark: cim-detect: not a matmul shape this pass offloads (linalg.conv_2d_nhwc_hwcf)
  %0 = linalg.conv_2d_nhwc_hwcf
       ins(%in, %filter : tensor<1x8x8x4xi8>, tensor<3x3x4x4xi8>)
       outs(%init : tensor<1x6x6x4xi32>) -> tensor<1x6x6x4xi32>
  return %0 : tensor<1x6x6x4xi32>
}

// -----

// Right op, wrong contract: an i8 accumulator cannot hold a matmul's
// partial sums, so this is not the INT8-in/wider-out shape v0.1 offloads.
func.func @narrow_accumulator_reports_the_contract_reason(
    %act: tensor<4x8xi8>, %init: tensor<4x16xi8>) -> tensor<4x16xi8> {
  %weights = arith.constant dense<1> : tensor<8x16xi8>
  // CHECK: remark: cim-detect: does not match the INT8-in/wider-accumulator-out
  %0 = linalg.matmul ins(%act, %weights : tensor<4x8xi8>, tensor<8x16xi8>)
                     outs(%init : tensor<4x16xi8>) -> tensor<4x16xi8>
  return %0 : tensor<4x16xi8>
}

// -----

// Right op, right contract, but BOTH operands are runtime values -- there
// is no constant weight to make resident, so weight-stationary hardware
// buys this nothing. The count is named in the message (0), which is what
// distinguishes this from the two-constants case.
func.func @no_constant_operand_reports_the_count(
    %act: tensor<4x8xi8>, %weights: tensor<8x16xi8>,
    %init: tensor<4x16xi32>) -> tensor<4x16xi32> {
  // CHECK: remark: cim-detect: needs exactly one constant (weight) operand, found 0
  %0 = linalg.matmul ins(%act, %weights : tensor<4x8xi8>, tensor<8x16xi8>)
                     outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// -----

// The "prove it doesn't complain about everything" case: a real candidate
// is annotated and must produce NO remark. Without it, a regression that
// emitted a decline remark unconditionally would still satisfy all three
// CHECKs above -- the COUNT below is what actually catches that.
func.func @a_real_candidate_is_silent(%act: tensor<4x8xi8>,
                                       %init: tensor<4x16xi32>)
    -> tensor<4x16xi32> {
  %weights = arith.constant dense<1> : tensor<8x16xi8>
  %0 = linalg.matmul ins(%act, %weights : tensor<4x8xi8>, tensor<8x16xi8>)
                     outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// Exactly three decline remarks across the four sections above -- one per
// declining section, none for the valid candidate. The counted directive
// plus the negative one below pin BOTH bounds: too few means a decline
// went silent again, too many means a real candidate started being
// reported as declined.
//
// (Those two directives are deliberately NOT named in this prose. FileCheck
// scans comment text too, so spelling a directive out in an explanation
// makes the explanation itself a directive -- which is how the first draft
// of this file failed, with "invalid count in -COUNT specification" pointing
// at a sentence rather than at a check.)
// TOTAL-COUNT-3: remark: cim-detect:
// TOTAL-NOT: remark: cim-detect:
