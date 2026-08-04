// RUN: cim-opt %s -split-input-file -verify-diagnostics
//
// The verifiers from spec Sec. 5.4 must actually reject bad IR. A verifier
// that only ever returns success is worse than none, because it implies a
// guarantee the compiler does not provide.

func.func @program_shape_mismatch(%tile: !cim.tile<256x256xi8>,
                                   %w: memref<128x128xi8, #cim.space<host>>)
    -> !cim.resident<256x256xi8> {
  // expected-error @+1 {{weights shape}}
  %r = cim.program %tile, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<256x256xi8>, memref<128x128xi8, #cim.space<host>>)
       -> !cim.resident<256x256xi8>
  return %r : !cim.resident<256x256xi8>
}

// -----

func.func @program_resident_geometry_mismatch(%tile: !cim.tile<256x256xi8>,
                                               %w: memref<256x256xi8, #cim.space<host>>)
    -> !cim.resident<128x128xi8> {
  // expected-error @+1 {{resident shape}}
  %r = cim.program %tile, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<256x256xi8>, memref<256x256xi8, #cim.space<host>>)
       -> !cim.resident<128x128xi8>
  return %r : !cim.resident<128x128xi8>
}

// -----

// Rule 1: activation length must match the resident weights' column count.
func.func @mvm_activation_length_mismatch(%r: !cim.resident<256x256xi8>,
                                           %act: memref<128xi8, #cim.space<near>>)
    -> memref<256xi32, #cim.space<near>> {
  // expected-error @+1 {{activations must be a 1-D memref of 256 elements}}
  %out = cim.mvm %r, %act
         : (!cim.resident<256x256xi8>, memref<128xi8, #cim.space<near>>)
         -> memref<256xi32, #cim.space<near>>
  return %out : memref<256xi32, #cim.space<near>>
}

// -----

// Partial sums must not silently saturate (spec Sec. 5.3): accumulating i8
// weights into an i8 result is exactly the bug the i32 accumulator exists
// to prevent.
func.func @mvm_accumulator_too_narrow(%r: !cim.resident<256x256xi8>,
                                       %act: memref<256xi8, #cim.space<near>>)
    -> memref<256xi8, #cim.space<near>> {
  // expected-error @+1 {{partial sums would saturate}}
  %out = cim.mvm %r, %act
         : (!cim.resident<256x256xi8>, memref<256xi8, #cim.space<near>>)
         -> memref<256xi8, #cim.space<near>>
  return %out : memref<256xi8, #cim.space<near>>
}

// -----

// Rule 4: reduce_partial operands must all have identical shape.
func.func @reduce_partial_shape_mismatch(%p0: memref<256xi32, #cim.space<near>>,
                                          %p1: memref<128xi32, #cim.space<near>>)
    -> memref<256xi32, #cim.space<near>> {
  // expected-error @+1 {{all partial sums must have identical shape}}
  %sum = cim.reduce_partial %p0, %p1
         : (memref<256xi32, #cim.space<near>>, memref<128xi32, #cim.space<near>>)
         -> memref<256xi32, #cim.space<near>>
  return %sum : memref<256xi32, #cim.space<near>>
}

// -----

// Rule 4: reducing narrow partial sums defeats the point of the i32
// accumulator.
func.func @reduce_partial_non_integer(%p0: memref<256xf32, #cim.space<near>>,
                                       %p1: memref<256xf32, #cim.space<near>>)
    -> memref<256xf32, #cim.space<near>> {
  // expected-error @+1 {{integer accumulator type}}
  %sum = cim.reduce_partial %p0, %p1
         : (memref<256xf32, #cim.space<near>>, memref<256xf32, #cim.space<near>>)
         -> memref<256xf32, #cim.space<near>>
  return %sum : memref<256xf32, #cim.space<near>>
}

// -----

// A copy between identical spaces is a no-op that almost certainly means
// the transfer-insertion pass got something wrong.
func.func @copy_same_space(%src: memref<256xi8, #cim.space<near>>)
    -> memref<256xi8, #cim.space<near>> {
  // expected-error @+1 {{same memory space}}
  %d = cim.copy %src : memref<256xi8, #cim.space<near>>
                    to memref<256xi8, #cim.space<near>>
  return %d : memref<256xi8, #cim.space<near>>
}

// -----

func.func @copy_changes_shape(%src: memref<256xi8, #cim.space<host>>)
    -> memref<128xi8, #cim.space<near>> {
  // expected-error @+1 {{must not change shape}}
  %d = cim.copy %src : memref<256xi8, #cim.space<host>>
                    to memref<128xi8, #cim.space<near>>
  return %d : memref<128xi8, #cim.space<near>>
}

// -----

// effective_bits models what the readout path can resolve; claiming more
// bits than the result type holds is a target-description error.
func.func @requantize_bits_exceed_result(%sum: memref<256xi32, #cim.space<near>>)
    -> memref<256xi8, #cim.space<near>> {
  // expected-error @+1 {{exceeds the width of the result type}}
  %q = cim.requantize %sum {scale = 1.0 : f32, zero_point = 0 : i32, effective_bits = 16 : i32}
       : memref<256xi32, #cim.space<near>> -> memref<256xi8, #cim.space<near>>
  return %q : memref<256xi8, #cim.space<near>>
}

// -----

// TileType::verify's branches were unreachable from parsing until
// TileType::parse switched from get() to getChecked(): get() asserts in a
// debug build and silently constructs a malformed type otherwise.
// expected-error @+1 {{cim.tile expects a 2D shape}}
func.func @tile_must_be_2d(%t: !cim.tile<256xi8>) {
  return
}

// -----

// expected-error @+1 {{cim.tile dimensions must be positive}}
func.func @tile_dims_must_be_positive(%t: !cim.tile<0x256xi8>) {
  return
}

// -----

// A tile holds integer weights. This verified until the element-type check
// was added -- the verifier took the argument and never looked at it, so a
// float tile produced structurally valid IR for a device that cannot hold
// one.
// expected-error @+1 {{cim.tile element type must be an integer type}}
func.func @tile_element_type_must_be_integer(%t: !cim.tile<4x4xf32>) {
  return
}
