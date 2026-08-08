// RUN: cim-opt %s --cim-legalize-precision=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file | FileCheck %s
// RUN: cim-opt %s --cim-legalize-precision=target-yaml=%S/../targets/tiny-4x4-4bit.yaml \
// RUN:   --split-input-file 2>&1 1>/dev/null | FileCheck --check-prefix=WARN %s
// RUN: not cim-opt %s --cim-legalize-precision --split-input-file 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-TARGET %s

// Pass 6 (spec Sec. 6). Most cases below hand-write IR shaped the way
// cim-partition's output IS shaped locally: a terminal cim.mvm/
// cim.reduce_partial result whose only other use is a bare memref.dealloc
// (nothing assuming its old type), which always narrows to i8. The two
// composition cases at the end exercise this pass against a downstream
// chain: one that IS safe to retype (a cim.copy into a locally-owned
// memref.alloc this pass can resize) and real cim-partition output (whose
// copy-back ultimately writes into the function's own output argument,
// which this pass cannot retype) -- see lib/Transforms/CIMLegalizePrecision.cpp's
// file header for why both are handled without ever mislowering.
// test/Transforms/cim-pipeline-full.mlir covers the full composed pipeline.

// NO-TARGET: requires -target-yaml

// -----

// A single-tile mvm result: nothing consumes it through a
// cim.reduce_partial, so it is itself the terminal accumulator and gets
// requantized directly. scale=1.0/zero_point=0 always (v0.1 has no
// calibration step to derive anything else from -- see the pass's header
// comment), and effective_bits comes straight from the target file.
// Existing uses (the memref.dealloc here) are rewired to read the
// requantize result, not the original i32 mvm result.
// CHECK-LABEL: func.func @single_tile_mvm_gets_requantized
func.func @single_tile_mvm_gets_requantized(%dev: !cim.device<"t">) {
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %act = memref.alloc() : memref<4xi8, #cim.space<near>>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
         -> memref<4xi32, #cim.space<near>>
  memref.dealloc %out : memref<4xi32, #cim.space<near>>
  // CHECK: %[[OUT:.*]] = cim.mvm
  // CHECK-NEXT: %[[RQ:.*]] = cim.requantize %[[OUT]]
  // CHECK-SAME: effective_bits = 8 : i32
  // CHECK-SAME: scale = 1.000000e+00 : f32
  // CHECK-SAME: zero_point = 0 : i32
  // CHECK-SAME: memref<4xi32, #cim.space<near>> -> memref<4xi8, #cim.space<near>>
  // CHECK-NEXT: memref.dealloc %[[RQ]]
  return
}

// -----

// A two-tile K-split: each cim.mvm feeds cim.reduce_partial, so each is an
// INTERMEDIATE partial sum -- not yet the value a consumer outside the CIM
// domain should see -- and must NOT be requantized. Only reduce_partial's
// own result is terminal.
// CHECK-LABEL: func.func @multi_tile_reduce_partial_gets_requantized
func.func @multi_tile_reduce_partial_gets_requantized(%dev: !cim.device<"t">) {
  %t0 = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %t1 = cim.tile_alloc %dev {id = 1 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w0 = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %w1 = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %r0 = cim.program %t0, %w0 {cost_ns = 1 : i64, cost_pj = 1 : i64}
        : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %r1 = cim.program %t1, %w1 {cost_ns = 1 : i64, cost_pj = 1 : i64}
        : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %act = memref.alloc() : memref<4xi8, #cim.space<near>>
  %p0 = cim.mvm %r0, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
        -> memref<4xi32, #cim.space<near>>
  %p1 = cim.mvm %r1, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
        -> memref<4xi32, #cim.space<near>>
  %sum = cim.reduce_partial %p0, %p1
         : (memref<4xi32, #cim.space<near>>, memref<4xi32, #cim.space<near>>) -> memref<4xi32, #cim.space<near>>
  memref.dealloc %sum : memref<4xi32, #cim.space<near>>
  // CHECK: %[[P0:.*]] = cim.mvm
  // CHECK: %[[P1:.*]] = cim.mvm
  // CHECK-NOT: cim.requantize %[[P0]]
  // CHECK-NOT: cim.requantize %[[P1]]
  // CHECK: %[[SUM:.*]] = cim.reduce_partial %[[P0]], %[[P1]]
  // CHECK-NEXT: %[[RQ:.*]] = cim.requantize %[[SUM]]
  // CHECK-NEXT: memref.dealloc %[[RQ]]
  return
}

// -----

// Idempotency: a terminal value that already has a cim.requantize reading
// it (e.g. because a caller's pipeline runs this pass twice) must not get a
// second one stacked on top -- alreadyLegalized exists for exactly this.
// CHECK-LABEL: func.func @already_legalized_terminal_is_left_alone
func.func @already_legalized_terminal_is_left_alone(%dev: !cim.device<"t">) {
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %act = memref.alloc() : memref<4xi8, #cim.space<near>>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
         -> memref<4xi32, #cim.space<near>>
  %rq = cim.requantize %out {scale = 1.0 : f32, zero_point = 0 : i32, effective_bits = 8 : i32}
        : memref<4xi32, #cim.space<near>> -> memref<4xi8, #cim.space<near>>
  memref.dealloc %rq : memref<4xi8, #cim.space<near>>
  // CHECK: cim.mvm
  // CHECK: cim.requantize
  // CHECK-NOT: cim.requantize
  return
}

// -----

// The warning path: an effective_bits < 8 target must warn with the
// estimated accuracy impact (spec Sec. 6) rather than silently degrading.
// Diagnostics only -- run with the module's stdout redirected away, see the
// WARN RUN line above.
// WARN: this target's precision.output_effective_bits (4) is less than 8
// WARN-SAME: 240 of the 256 encodings
//
// CHECK-LABEL bounds the previous chunk's CHECK-NOT to this function's own
// output rather than letting it run to end-of-file, where this chunk's own
// (unrelated) cim.requantize would otherwise trip it under the main RUN
// line, which applies to every chunk here, not just the WARN one.
// CHECK-LABEL: func.func @sub_8_bit_target_warns
func.func @sub_8_bit_target_warns(%dev: !cim.device<"t">) {
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %act = memref.alloc() : memref<4xi8, #cim.space<near>>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
         -> memref<4xi32, #cim.space<near>>
  memref.dealloc %out : memref<4xi32, #cim.space<near>>
  return
}

// -----

// A terminal whose copy-back chain is entirely local (a cim.copy to host
// followed by a memref.copy into a memref.alloc this pass owns, mirroring
// the shape cim-partition emits but with the copy landing in scratch this
// pass is free to resize instead of a function argument) narrows all the
// way to i8, and BOTH downstream ops are retyped to match -- not just the
// inserted cim.requantize.
// CHECK-LABEL: func.func @narrows_through_a_locally_owned_copy_chain
func.func @narrows_through_a_locally_owned_copy_chain(%dev: !cim.device<"t">) {
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %act = memref.alloc() : memref<4xi8, #cim.space<near>>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
         -> memref<4xi32, #cim.space<near>>
  %host = cim.copy %out : memref<4xi32, #cim.space<near>> to memref<4xi32>
  %scratch = memref.alloc() : memref<4xi32>
  memref.copy %host, %scratch : memref<4xi32> to memref<4xi32>
  memref.dealloc %scratch : memref<4xi32>
  // CHECK: %[[OUT:.*]] = cim.mvm
  // CHECK-NEXT: %[[RQ:.*]] = cim.requantize %[[OUT]]
  // CHECK-SAME: memref<4xi32, #cim.space<near>> -> memref<4xi8, #cim.space<near>>
  // CHECK-NEXT: %[[HOST:.*]] = cim.copy %[[RQ]]
  // CHECK-SAME: memref<4xi8, #cim.space<near>> to memref<4xi8>
  // CHECK-NEXT: %[[SCRATCH:.*]] = memref.alloc() : memref<4xi8>
  // CHECK-NEXT: memref.copy %[[HOST]], %[[SCRATCH]] : memref<4xi8> to memref<4xi8>
  // CHECK-NEXT: memref.dealloc %[[SCRATCH]] : memref<4xi8>
  return
}

// -----

// A terminal whose copy-back chain bottoms out at a value this pass does
// not own (here: a plain memref.alloc-typed function ARGUMENT rather than
// scratch cim-legalize-precision itself allocated -- unlike the previous
// case, this pass has no license to resize somebody else's buffer) falls
// back to a requantize whose result stays the terminal's ORIGINAL element
// type: the value is still genuinely clamped to effective_bits (the
// interpreter's runRequantize enforces that regardless of container
// width), but nothing downstream needs retyping because nothing changed
// type. This is the shape real cim-partition output takes (a copy-back
// into a subview of the function's own output argument), reproduced here
// with a plain argument standing in for that subview.
// CHECK-LABEL: func.func @falls_back_when_the_sink_is_not_owned
func.func @falls_back_when_the_sink_is_not_owned(%dev: !cim.device<"t">, %sink: memref<4xi32>) {
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.alloc() : memref<4x4xi8, #cim.space<host>>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8, #cim.space<host>>) -> !cim.resident<4x4xi8>
  %act = memref.alloc() : memref<4xi8, #cim.space<near>>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>)
         -> memref<4xi32, #cim.space<near>>
  %host = cim.copy %out : memref<4xi32, #cim.space<near>> to memref<4xi32>
  memref.copy %host, %sink : memref<4xi32> to memref<4xi32>
  // CHECK: %[[OUT:.*]] = cim.mvm
  // CHECK-NEXT: %[[RQ:.*]] = cim.requantize %[[OUT]]
  // CHECK-SAME: memref<4xi32, #cim.space<near>> -> memref<4xi32, #cim.space<near>>
  // CHECK-NEXT: %[[HOST:.*]] = cim.copy %[[RQ]]
  // CHECK-SAME: memref<4xi32, #cim.space<near>> to memref<4xi32>
  // CHECK-NEXT: memref.copy %[[HOST]], %{{.*}} : memref<4xi32> to memref<4xi32>
  return
}
