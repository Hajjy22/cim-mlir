// RUN: cim-opt %s --cim-detect --split-input-file | FileCheck %s
//
// Pass 1 (spec Sec. 6) annotates eligible matmuls and converts nothing.
// The negative cases below matter as much as the positive one: a detector
// that annotates too eagerly hands cim-partition work it cannot lower.

// An INT8 matmul against a constant weight is the v0.1 contract exactly.
// CHECK-LABEL: func.func @int8_matmul_with_constant_weight
func.func @int8_matmul_with_constant_weight(%act: tensor<4x8xi8>,
                                             %init: tensor<4x16xi32>)
    -> tensor<4x16xi32> {
  %weights = arith.constant dense<1> : tensor<8x16xi8>
  // CHECK: linalg.matmul
  // CHECK-SAME: cim.candidate
  // CHECK-SAME: cim.weight_operand = 1
  %0 = linalg.matmul ins(%act, %weights : tensor<4x8xi8>, tensor<8x16xi8>)
                     outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// -----

// Floating point is out of scope for v0.1 (spec Sec. 2). Annotating it
// would hand cim-partition a matmul the hardware model cannot express.
// CHECK-LABEL: func.func @f32_matmul_is_not_a_candidate
func.func @f32_matmul_is_not_a_candidate(%act: tensor<4x8xf32>,
                                          %init: tensor<4x16xf32>)
    -> tensor<4x16xf32> {
  %weights = arith.constant dense<1.0> : tensor<8x16xf32>
  // CHECK-NOT: cim.candidate
  %0 = linalg.matmul ins(%act, %weights : tensor<4x8xf32>, tensor<8x16xf32>)
                     outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

// Two activations: there is no weight to hold resident, so weight-stationary
// hardware buys this nothing.
// CHECK-LABEL: func.func @activation_times_activation_is_not_a_candidate
func.func @activation_times_activation_is_not_a_candidate(%a: tensor<4x8xi8>,
                                                           %b: tensor<8x16xi8>,
                                                           %init: tensor<4x16xi32>)
    -> tensor<4x16xi32> {
  // CHECK-NOT: cim.candidate
  %0 = linalg.matmul ins(%a, %b : tensor<4x8xi8>, tensor<8x16xi8>)
                     outs(%init : tensor<4x16xi32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}

// -----

// An i8 accumulator would saturate partial sums -- the exact bug the i32
// accumulator exists to prevent, and what cim.mvm's verifier rejects.
// CHECK-LABEL: func.func @narrow_accumulator_is_not_a_candidate
func.func @narrow_accumulator_is_not_a_candidate(%act: tensor<4x8xi8>,
                                                  %init: tensor<4x16xi8>)
    -> tensor<4x16xi8> {
  %weights = arith.constant dense<1> : tensor<8x16xi8>
  // CHECK-NOT: cim.candidate
  %0 = linalg.matmul ins(%act, %weights : tensor<4x8xi8>, tensor<8x16xi8>)
                     outs(%init : tensor<4x16xi8>) -> tensor<4x16xi8>
  return %0 : tensor<4x16xi8>
}

// -----

// Convolution is explicitly out of scope for v0.1 and must not be picked up
// just because its element types happen to match.
// CHECK-LABEL: func.func @conv_is_not_a_candidate
func.func @conv_is_not_a_candidate(%in: tensor<1x8x8x4xi8>,
                                    %init: tensor<1x6x6x4xi32>)
    -> tensor<1x6x6x4xi32> {
  %filter = arith.constant dense<1> : tensor<3x3x4x4xi8>
  // CHECK-NOT: cim.candidate
  %0 = linalg.conv_2d_nhwc_hwcf
       ins(%in, %filter : tensor<1x8x8x4xi8>, tensor<3x3x4x4xi8>)
       outs(%init : tensor<1x6x6x4xi32>) -> tensor<1x6x6x4xi32>
  return %0 : tensor<1x6x6x4xi32>
}

// -----

// Batched matmul is in scope for v0.1.
// CHECK-LABEL: func.func @batch_matmul_is_a_candidate
func.func @batch_matmul_is_a_candidate(%act: tensor<2x4x8xi8>,
                                        %init: tensor<2x4x16xi32>)
    -> tensor<2x4x16xi32> {
  %weights = arith.constant dense<1> : tensor<2x8x16xi8>
  // CHECK: linalg.batch_matmul
  // CHECK-SAME: cim.candidate
  %0 = linalg.batch_matmul ins(%act, %weights : tensor<2x4x8xi8>, tensor<2x8x16xi8>)
                           outs(%init : tensor<2x4x16xi32>) -> tensor<2x4x16xi32>
  return %0 : tensor<2x4x16xi32>
}
