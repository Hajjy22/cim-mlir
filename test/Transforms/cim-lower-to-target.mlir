// RUN: cim-opt %s --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file --verify-diagnostics | FileCheck %s
// RUN: not cim-opt %s --cim-lower-to-target --split-input-file 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-TARGET %s

// Pass 7 (spec Sec. 6), v0.1 scope: straight-line, single-tile code only.
// See lib/Transforms/CIMLowerToTarget.cpp's file header for the full design
// note (memory model, freeing rules, error handling). Manually verified
// once beyond what these structural checks cover: the pass's output for a
// case shaped like @straight_line_lowers_every_op below was taken all the
// way through MLIR's standard --convert-to-llvm pipeline, mlir-translate,
// clang, and linked against the real runtime/libcimrt.a, then actually RUN
// as a native binary -- it computed the correct cim.mvm result against real
// (simulated) hardware, not the interpreter. That is not automated here
// (it needs a linker and a target triple this suite has no business
// depending on); this file is the structural half of verification, the
// same two-tier split every other pass in this project uses.

// NO-TARGET: requires -target-yaml

// -----

// Every real op this pass lowers, once each, in the order cim-partition's
// own output would produce them: open, allocate a tile, program it, run an
// mvm, synchronize, and free the result. Checks the call sequence and that
// every status is checked (cf.assert immediately follows every call that
// returns one).
// CHECK-LABEL: func.func @straight_line_lowers_every_op
func.func @straight_line_lowers_every_op(%act: memref<4xi8>) {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.get_global @w : memref<4x4xi8>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
  cim.barrier %dev : !cim.device<"t">
  memref.dealloc %out : memref<4xi32>
  return
}
// CHECK: %[[STATUS0:.*]] = call @cimrt_open
// CHECK: cf.assert
// device_open's out-param is loaded straight back into a device value with
// no cim.tile_alloc call left in the IR -- a tile id needs no runtime call
// of its own (see the file header's tile/resident note).
// CHECK-NOT: cim.tile_alloc
// CHECK: call @cimrt_alloc
// CHECK: cf.assert
// CHECK: call @cimrt_write
// CHECK: cf.assert
// CHECK: call @cimrt_program
// CHECK: cf.assert
// CHECK: call @cimrt_free
// The activation is staged (it is still a real host memref here, no
// cim-insert-transfers having run) before the mvm call.
// CHECK: call @cimrt_alloc
// CHECK: cf.assert
// CHECK: call @cimrt_write
// CHECK: cf.assert
// CHECK: call @cimrt_alloc
// CHECK: cf.assert
// CHECK: call @cimrt_mvm
// CHECK: cf.assert
// The mvm's result is host-declared (no #cim.space), so it is read back
// into a real, freshly allocated memref rather than staying an opaque
// handle.
// CHECK: call @cimrt_read
// CHECK: cf.assert
// CHECK: call @cimrt_barrier
// CHECK: cf.assert
// CHECK-NOT: cim.device_open
// CHECK-NOT: cim.program
// CHECK-NOT: cim.mvm
// CHECK-NOT: cim.barrier
memref.global "private" constant @w : memref<4x4xi8> = dense<1>

// -----

// A device-space cim.mvm result (#cim.space<near>) stays an opaque
// !llvm.ptr handle -- no read-back -- and a later memref.dealloc on it
// becomes cimrt_free rather than an ordinary memref.dealloc.
// CHECK-LABEL: func.func @dealloc_on_device_space_result_becomes_cimrt_free
func.func @dealloc_on_device_space_result_becomes_cimrt_free() {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.alloc() : memref<4x4xi8>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  %act = memref.alloc() : memref<4xi8>
  %out = cim.mvm %r, %act : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32, #cim.space<near>>
  memref.dealloc %out : memref<4xi32, #cim.space<near>>
  return
}
// CHECK: %[[OUT:.*]] = call @cimrt_mvm
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.dealloc
// The activation staging buffer is freed automatically (scratch this pass
// allocated for the one mvm call), and the mvm's own device-space result
// is freed only because of the memref.dealloc actually present in the
// source -- two frees, neither one an ordinary memref.dealloc.
// CHECK: call @cimrt_free
// CHECK: call @cimrt_free
// CHECK-NEXT: return

// -----

// cim.copy, host source -> device dest: allocates and writes a fresh
// buffer, does NOT read back (the result stays a handle).
// CHECK-LABEL: func.func @copy_host_to_device_stages_and_writes
func.func @copy_host_to_device_stages_and_writes() {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %src = memref.alloc() : memref<4xi8>
  %dst = cim.copy %src : memref<4xi8> to memref<4xi8, #cim.space<near>>
  memref.dealloc %dst : memref<4xi8, #cim.space<near>>
  return
}
// CHECK: call @cimrt_alloc
// CHECK: cf.assert
// CHECK: call @cimrt_write
// CHECK: cf.assert
// CHECK-NOT: call @cimrt_read
// CHECK: call @cimrt_free

// -----

// cim.copy, device source -> host dest: allocates a REAL destination
// memref and reads the device buffer's bytes into it.
// CHECK-LABEL: func.func @copy_device_to_host_reads_back
func.func @copy_device_to_host_reads_back() {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %src = memref.alloc() : memref<4xi8>
  %near = cim.copy %src : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %host = cim.copy %near : memref<4xi8, #cim.space<near>> to memref<4xi8>
  memref.dealloc %host : memref<4xi8>
  return
}
// CHECK: %[[NEAR:.*]] = call @cimrt_alloc
// CHECK: %[[HOSTALLOC:.*]] = memref.alloc() : memref<4xi8>
// CHECK: call @cimrt_read
// CHECK: cf.assert
// The real destination memref is what the ordinary, untouched
// memref.dealloc below now targets -- not cimrt_free.
// CHECK: memref.dealloc %[[HOSTALLOC]]

// -----

// cim.copy, device source -> device dest (near -> insitu): cimrt_copy
// moves buffer to buffer directly, no host pointer, no read/write.
// CHECK-LABEL: func.func @copy_device_to_device_uses_cimrt_copy
func.func @copy_device_to_device_uses_cimrt_copy() {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %src = memref.alloc() : memref<4xi8>
  %near = cim.copy %src : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %insitu = cim.copy %near : memref<4xi8, #cim.space<near>> to memref<4xi8, #cim.space<insitu>>
  memref.dealloc %insitu : memref<4xi8, #cim.space<insitu>>
  return
}
// CHECK: call @cimrt_copy
// CHECK: cf.assert
// CHECK-NOT: call @cimrt_write
// CHECK-NOT: call @cimrt_read

// -----

// cim.copy, host to host (an unspaced source, an explicitly #cim.space<host>
// dest -- CopyOp::verify() allows this since the two Attributes are not
// identical, see the pass' own file header): no cimrt involvement, and no
// device needed at all.
// CHECK-LABEL: func.func @copy_host_to_host_uses_plain_memref_copy
func.func @copy_host_to_host_uses_plain_memref_copy(%src: memref<4xi8>) {
  %dst = cim.copy %src : memref<4xi8> to memref<4xi8, #cim.space<host>>
  memref.dealloc %dst : memref<4xi8, #cim.space<host>>
  return
}
// CHECK-NOT: cim.device_open
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4xi8, #cim.space<host>>
// CHECK-NEXT: memref.copy %{{.*}}, %[[ALLOC]]
// CHECK-NEXT: memref.dealloc %[[ALLOC]]
// CHECK-NOT: call @cimrt_

// -----

// An IDENTITY memref.subview (offset 0, full extent, unit stride) of a
// device-space value folds straight through to the same handle -- exactly
// the shape cim-partition emits slicing a staged activation for a single
// K-tile (test/Transforms/cim-pipeline-full.mlir exercises this against
// real cim-partition output). No cimrt call is needed for the subview
// itself, and cim.mvm reads the very same pointer cimrt_alloc/cimrt_write
// produced for the copy, not a second buffer.
// CHECK-LABEL: func.func @identity_subview_of_a_device_value_folds
func.func @identity_subview_of_a_device_value_folds(%act: memref<4xi8>) {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.get_global @w2 : memref<4x4xi8>
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  %near = cim.copy %act : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %slice = memref.subview %near[0] [4] [1]
           : memref<4xi8, #cim.space<near>> to memref<4xi8, strided<[1]>, #cim.space<near>>
  %out = cim.mvm %r, %slice : (!cim.resident<4x4xi8>, memref<4xi8, strided<[1]>, #cim.space<near>>) -> memref<4xi32>
  memref.dealloc %out : memref<4xi32>
  return
}
// The weight is staged and programmed first (cim.program precedes cim.copy
// in program order here), then the activation is staged into its own
// buffer -- %[[ACTBUF]] is the SAME value cimrt_mvm below reads from,
// captured right after the one cimrt_alloc/cimrt_write pair the copy
// causes.
// CHECK: call @cimrt_program
// CHECK: %[[ACTPTR:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr
// CHECK: call @cimrt_write(%[[ACTPTR]]
// CHECK-NOT: memref.subview
// CHECK: call @cimrt_mvm(%{{.*}}, %{{.*}}, %[[ACTPTR]]
// CHECK-NOT: cim.mvm
memref.global "private" constant @w2 : memref<4x4xi8> = dense<1>

// -----

// A NON-identity, rank-1, unit-stride memref.subview (a real slice, offset
// nonzero) of a device-space value is MATERIALIZED into a fresh buffer via
// cimrt_copy_range -- this is the real multi-K-tile case (cim-partition's
// subView1D slicing a staged activation per K-tile once there is more than
// one), and closing it is what lets a real multi-K-tile matmul reach
// cimrt_mvm through this pass at all (test/Transforms/cim-partition.mlir's
// output for such a matmul, run through this pass, no longer trips this
// case -- see cim-pipeline-full.mlir's own MULTI_K run line).
// CHECK-LABEL: func.func @non_identity_rank1_slice_is_materialized
func.func @non_identity_rank1_slice_is_materialized(%act: memref<8xi8>) {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %near = cim.copy %act : memref<8xi8> to memref<8xi8, #cim.space<near>>
  // CHECK: call @cimrt_write(%[[SRC:[0-9]+]],
  %slice = memref.subview %near[4] [4] [1]
           : memref<8xi8, #cim.space<near>> to memref<4xi8, strided<[1], offset: 4>, #cim.space<near>>
  // A fresh 4-byte buffer, filled from byte offset 4 of the 8-byte source
  // -- the slice's own offset and size, not the identity case's whole
  // buffer.
  // CHECK: call @cimrt_alloc
  // CHECK: %[[SLICEDPTR:.*]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[DSTOFF:.*]] = arith.constant 0 : i64
  // CHECK: %[[SRCOFF:.*]] = arith.constant 4 : i64
  // CHECK: %[[LEN:.*]] = arith.constant 4 : i64
  // CHECK: call @cimrt_copy_range(%[[SLICEDPTR]], %[[DSTOFF]], %[[SRC]], %[[SRCOFF]], %[[LEN]])
  // CHECK-NOT: error
  memref.dealloc %slice : memref<4xi8, strided<[1], offset: 4>, #cim.space<near>>
  // The dealloc frees the MATERIALIZED buffer, not the original source.
  // CHECK: call @cimrt_free(%[[SLICEDPTR]])
  return
}

// -----

// A non-identity slice of a RANK-2 (or higher) device-space source is still
// refused: rank1ContiguousSliceByteRange only materializes a rank-1 slice
// of a rank-1 source, matching cim-partition's actual subView1D output --
// a genuine multi-dimensional slice is a different, still-open ABI
// question (see the M4 roadmap entry).
func.func @non_identity_higher_rank_slice_is_still_refused(%act: memref<4x8xi8>) {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %near = cim.copy %act : memref<4x8xi8> to memref<4x8xi8, #cim.space<near>>
  // expected-error @+1 {{neither an identity slice}}
  %slice = memref.subview %near[0, 4] [4, 4] [1, 1]
           : memref<4x8xi8, #cim.space<near>> to memref<4x4xi8, strided<[8, 1], offset: 4>, #cim.space<near>>
  memref.dealloc %slice : memref<4x4xi8, strided<[8, 1], offset: 4>, #cim.space<near>>
  return
}

// -----

// A tile id outside the target's declared tile count is refused at compile
// time against -target-yaml, not deferred to a runtime cimrt_query.
// tiny-4x4.yaml declares 2 tiles (ids 0 and 1); id 5 is out of range.
func.func @tile_id_out_of_range() {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  // expected-error @+1 {{tile id 5 is outside the target's 2 tiles}}
  %t = cim.tile_alloc %dev {id = 5 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  cim.tile_free %t : !cim.tile<4x4xi8>
  return
}

// -----

// cim.reduce_partial lowers against cimrt_reduce_add. Two independently-
// staged partials (not sliced from one shared activation buffer -- that
// shape needs non-identity device-space subview lowering, a separate,
// still-open scope limit; see checkAllowedConsumers' own comment), each
// its own tile/program/mvm, reduced into one near-space result that stays
// a live device handle until a later cim.copy reads it back -- exactly
// the "device-space result, checked before rewriting" shape lowerMvm's
// own device branch already uses.
// CHECK-LABEL: func.func @reduce_two_partials_stays_device_until_copied
func.func @reduce_two_partials_stays_device_until_copied(
    %act0: memref<4xi8>, %act1: memref<4xi8>,
    %w0: memref<4x4xi8>, %w1: memref<4x4xi8>) -> memref<4xi32> {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %a0 = cim.copy %act0 : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %a1 = cim.copy %act1 : memref<4xi8> to memref<4xi8, #cim.space<near>>
  %t0 = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %r0 = cim.program %t0, %w0 {cost_ns = 1 : i64, cost_pj = 1 : i64, persistent = true}
        : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  %p0 = cim.mvm %r0, %a0 {accumulate = false}
        : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>) -> memref<4xi32, #cim.space<near>>
  %t1 = cim.tile_alloc %dev {id = 1 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %r1 = cim.program %t1, %w1 {cost_ns = 1 : i64, cost_pj = 1 : i64, persistent = true}
        : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  // CHECK: call @cimrt_program
  // CHECK: call @cimrt_mvm
  %p1 = cim.mvm %r1, %a1 {accumulate = false}
        : (!cim.resident<4x4xi8>, memref<4xi8, #cim.space<near>>) -> memref<4xi32, #cim.space<near>>
  // CHECK: call @cimrt_program
  // CHECK: call @cimrt_mvm
  %s = cim.reduce_partial %p0, %p1
       : (memref<4xi32, #cim.space<near>>, memref<4xi32, #cim.space<near>>) -> memref<4xi32, #cim.space<near>>
  // A single call for two operands: N-1 chained adds, N=2. Neither mvm
  // result is this pass' own scratch, so neither is freed around the call
  // -- only the fresh sum buffer cimrt_alloc just produced (this call's
  // own out argument) is a candidate for that, and it must not be freed
  // here either (it survives as the reduction's own result, checked
  // below).
  // CHECK: call @cimrt_reduce_add(%{{[0-9]+}}, %[[SUMBUF:[0-9]+]],
  // CHECK-NOT: call @cimrt_reduce_add
  // CHECK-NOT: call @cimrt_free(%[[SUMBUF]])
  %h = cim.copy %s : memref<4xi32, #cim.space<near>> to memref<4xi32>
  // The reduction's own result stays a live device handle straight through
  // to this copy's read-back -- no extra alloc/free pair in between.
  // CHECK: call @cimrt_read(%[[SUMBUF]],
  // CHECK-NOT: error
  return %h : memref<4xi32>
}

// -----

// Three operands chain exactly two calls (N-1), and every INTERMEDIATE
// accumulator this pass allocates to do so is freed once consumed -- only
// the final one survives, matching lowerMvm/lowerRequantize's own scratch
// discipline. All three partials are real host memrefs here (an
// unrealistic shape for real cim-partition output, which always produces
// near-space partials, but a legitimate one this pass must still handle
// correctly, and a stronger test of stageForRead's own staging path than
// the near-space case above exercises).
// CHECK-LABEL: func.func @reduce_three_partials_chains_two_calls
func.func @reduce_three_partials_chains_two_calls(
    %h0: memref<4xi32>, %h1: memref<4xi32>, %h2: memref<4xi32>) -> memref<4xi32> {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %p0 = cim.copy %h0 : memref<4xi32> to memref<4xi32, #cim.space<near>>
  %p1 = cim.copy %h1 : memref<4xi32> to memref<4xi32, #cim.space<near>>
  %p2 = cim.copy %h2 : memref<4xi32> to memref<4xi32, #cim.space<near>>
  // CHECK: call @cimrt_write(%[[P0BUF:[0-9]+]],
  // CHECK: call @cimrt_write(%[[P1BUF:[0-9]+]],
  // CHECK: call @cimrt_write(%[[P2BUF:[0-9]+]],
  %s = cim.reduce_partial %p0, %p1, %p2
       : (memref<4xi32, #cim.space<near>>, memref<4xi32, #cim.space<near>>,
          memref<4xi32, #cim.space<near>>) -> memref<4xi32>
  // CHECK: call @cimrt_reduce_add(%{{[0-9]+}}, %[[ACC1:[0-9]+]], %[[P0BUF]], %[[P1BUF]],
  // CHECK: call @cimrt_reduce_add(%{{[0-9]+}}, %[[ACC2:[0-9]+]], %[[ACC1]], %[[P2BUF]],
  // The first (intermediate) accumulator is freed once consumed; the
  // original partials are not this pass' scratch and are never freed here.
  // CHECK: call @cimrt_free(%[[ACC1]])
  // A host-declared result: read back and freed, like cimrt_mvm's own
  // host-result path.
  // CHECK: call @cimrt_read(%[[ACC2]],
  // CHECK: call @cimrt_free(%[[ACC2]])
  // CHECK-NOT: error
  return %s : memref<4xi32>
}

// -----

// cim.reduce_partial carries no device operand of its own in the dialect
// (same as cim.copy/cim.requantize); with no cim.device_open lowered
// anywhere in the function, there is no device to stage the reduction
// through.
func.func @reduce_partial_needs_a_device(%p0: memref<4xi32>, %p1: memref<4xi32>) -> memref<4xi32> {
  // expected-error @+1 {{cim.reduce_partial needs a device to stage through}}
  %s = cim.reduce_partial %p0, %p1 : (memref<4xi32>, memref<4xi32>) -> memref<4xi32>
  return %s : memref<4xi32>
}

// -----

// cim.requantize, host in and out: staged into a scratch near-space buffer
// (freed once its one use is done, same as cim.mvm's activation staging),
// requantized via cimrt_requantize, then read back into a real host
// memref for whatever consumes it -- the exact cim.mvm host-result shape,
// see @dealloc_on_device_space_result_becomes_cimrt_free's host case above
// for the same pattern.
// CHECK-LABEL: func.func @requantize_host_narrows
func.func @requantize_host_narrows(%in: memref<4xi32>) {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %q = cim.requantize %in {scale = 1.0 : f32, zero_point = 0 : i32, effective_bits = 8 : i32}
       : memref<4xi32> -> memref<4xi8>
  memref.dealloc %q : memref<4xi8>
  return
}
// CHECK: call @cimrt_open
// CHECK: call @cimrt_write(%[[INBUF:[0-9]+]],
// CHECK: cf.assert
// The exact operand order cimrt_requantize documents in cimrt.h: dev, in,
// out, count, in_bits, out_bits, scale, zero_point, effective_bits -- only
// the buffer identities (in, out) are pinned by name here, not the
// scalar constants, which the unit-level cimrt_requantize tests
// (test/unit/cimrt_test.cpp) and the numerical e2e test already cover.
// CHECK: call @cimrt_requantize(%{{.*}}, %[[INBUF]], %[[OUTBUF:[0-9]+]]
// CHECK: cf.assert
// The input scratch buffer is freed once its one use is done, exactly
// like cim.mvm's staged activation.
// CHECK: call @cimrt_free(%[[INBUF]])
// CHECK: %[[REALALLOC:.*]] = memref.alloc() : memref<4xi8>
// CHECK: call @cimrt_read(%[[OUTBUF]]
// CHECK: cf.assert
// CHECK: call @cimrt_free(%[[OUTBUF]])
// CHECK: memref.dealloc %[[REALALLOC]]
// CHECK-NOT: cim.requantize

// -----

// cim.requantize, device in and out (near -> near, the shape
// cim-legalize-precision's width-preserving fallback and its narrowing
// path both produce on real cim-partition output): no read-back, stays an
// opaque handle. The input is already ptrTy (staged by the preceding
// cim.copy this pass already lowered) -- deviceValueElemBits is what
// recovers its element width with no extra staging, so there is only ONE
// cimrt_alloc for the input (the copy's own), not a second one here.
// CHECK-LABEL: func.func @requantize_device_narrows_and_stays_a_handle
func.func @requantize_device_narrows_and_stays_a_handle(%act: memref<4xi32>) {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %near = cim.copy %act : memref<4xi32> to memref<4xi32, #cim.space<near>>
  %q = cim.requantize %near {scale = 1.0 : f32, zero_point = 0 : i32, effective_bits = 8 : i32}
       : memref<4xi32, #cim.space<near>> -> memref<4xi8, #cim.space<near>>
  memref.dealloc %q : memref<4xi8, #cim.space<near>>
  return
}
// CHECK: call @cimrt_open
// CHECK: call @cimrt_write(%[[COPYBUF:[0-9]+]],
// CHECK: cf.assert
// Reusing the SAME %[[COPYBUF]] as requantize's own input operand (not a
// freshly allocated one) is the proof that deviceValueElemBits recovered
// its width with no extra staging -- if a second buffer had been
// allocated and written instead, this capture would simply fail to match.
// CHECK: call @cimrt_requantize(%{{.*}}, %[[COPYBUF]], %[[OUTBUF2:[0-9]+]]
// CHECK: cf.assert
// The copy's own buffer has no explicit memref.dealloc in the source, so
// it leaks (file header point 4) -- only the requantize's OWN result,
// which the source's memref.dealloc explicitly targets, is freed.
// CHECK-NOT: call @cimrt_free(%[[COPYBUF]])
// CHECK: call @cimrt_free(%[[OUTBUF2]])
// CHECK-NOT: cim.requantize
// CHECK-NOT: cim.copy

// -----

// A cim op nested inside a region (an scf.for body -- exactly what
// cim-placement's loop hoisting produces) has no lowering yet.
func.func @op_inside_a_loop_is_refused(%dev: !cim.device<"t">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c3 step %c1 {
    // expected-error @+1 {{only lowers straight-line code}}
    cim.barrier %dev : !cim.device<"t">
  }
  return
}
