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
