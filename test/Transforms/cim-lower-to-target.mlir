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

// A NON-identity memref.subview (a real slice, offset nonzero) of a
// device-space value is refused -- cimrt_mvm has no offset/sub-buffer
// concept, so folding it the way the identity case does would silently
// read the wrong bytes. This is the real multi-K-tile case, deferred to
// M4 (docs/roadmap.md), not mislowered here.
func.func @non_identity_subview_is_refused(%act: memref<8xi8>) {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %near = cim.copy %act : memref<8xi8> to memref<8xi8, #cim.space<near>>
  // expected-error @+1 {{not an identity slice}}
  %slice = memref.subview %near[4] [4] [1]
           : memref<8xi8, #cim.space<near>> to memref<4xi8, strided<[1], offset: 4>, #cim.space<near>>
  memref.dealloc %slice : memref<4xi8, strided<[1], offset: 4>, #cim.space<near>>
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

// cim.reduce_partial is refused outright -- v0.1 scope is single-tile only.
func.func @reduce_partial_is_refused(%p0: memref<4xi32>, %p1: memref<4xi32>) {
  // expected-error @+1 {{cim.reduce_partial is not lowered yet}}
  %s = cim.reduce_partial %p0, %p1 : (memref<4xi32>, memref<4xi32>) -> memref<4xi32>
  memref.dealloc %s : memref<4xi32>
  return
}

// -----

// cim.requantize is refused outright, same reasoning.
func.func @requantize_is_refused(%in: memref<4xi32>) {
  // expected-error @+1 {{cim.requantize is not lowered yet}}
  %q = cim.requantize %in {scale = 1.0 : f32, zero_point = 0 : i32, effective_bits = 8 : i32}
       : memref<4xi32> -> memref<4xi8>
  memref.dealloc %q : memref<4xi8>
  return
}

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
