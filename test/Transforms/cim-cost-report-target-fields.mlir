// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-digital-cim.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-digital-cim.yaml \
// RUN:   --cim-cost-report=target-yaml=%S/../targets/tiny-digital-cim.yaml \
// RUN:   2>/dev/null | FileCheck %s

// WHY A SECOND TARGET, AND WHY THIS ONE
// =====================================
// cim-cost-report.mlir -- the file that pins this pass's whole JSON shape --
// runs exclusively against tiny-4x4.yaml, and tiny-4x4.yaml happens to
// declare `persistent: true`, `standby_leakage_uw_per_tile: 0.0` and
// `double_buffer_program: false`. Those are EXACTLY the CostReport struct's
// own default values, so for that one target a field the pass forgets to
// populate is indistinguishable from a field it populates correctly.
//
// The pass did forget three of them (numTiles, standbyLeakageUwPerTile,
// doubleBufferCapable) plus steadyStateElapsedNsPerInference, and the
// published JSON therefore described every target in the world as a
// leak-free, non-double-bufferable device -- while cim-bench, reading the
// same values straight from the spec, reported them correctly. Two emitters
// of one schema, disagreeing, with no test able to see it.
//
// tiny-digital-cim.yaml is the fixture that can see it: it declares
// `persistent: false`, `standby_leakage_uw_per_tile: 4.5` and
// `double_buffer_program: true` -- all three differing from the struct
// default -- so every CHECK below fails against the unfixed pass.

memref.global "private" constant @w : memref<4x4xi8> = dense<1>
memref.global "private" constant @a : memref<1x4xi8> = dense<2>

func.func private @sink(memref<1x4xi32>)

func.func @straight_line() {
  %w = memref.get_global @w : memref<4x4xi8>
  %aInit = memref.get_global @a : memref<1x4xi8>
  %a = memref.alloc() : memref<1x4xi8>
  memref.copy %aInit, %a : memref<1x4xi8> to memref<1x4xi8>
  %out = memref.alloc() : memref<1x4xi32>
  linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<4x4xi8>)
    outs(%out : memref<1x4xi32>)
  func.call @sink(%out) : (memref<1x4xi32>) -> ()
  return
}

// The target says volatile, so `persistent` must follow it rather than the
// struct's own `false`... which is also `false`. Checked anyway: it is the
// field the leakage floor below is gated on, so a regression that flipped
// it would silently zero the floor for a reason unrelated to the three
// fields this file exists for.
// CHECK: "persistent": false,

// Straight from the target file. Zero here means the pass dropped it.
// CHECK: "standby_leakage_uw_per_tile": 4.5,

// capabilities.double_buffer_program. This is the field that decides
// whether the overlap PROJECTION below is emitted at all, so `false` here
// does not merely misreport a capability -- it suppresses a whole output.
// CHECK: "double_buffer_capable": true,

// Emitted only because double_buffer_capable is true. A non-zero value
// also proves steadyStateElapsedNsPerInference itself got populated: the
// projection is max(program latency, mvm latency) and both terms come from
// the same arithmetic the elapsed figure does.
// CHECK: "steady_state_elapsed_ns_per_inference_if_overlapped": 120,

// The number that needs ALL of numTiles, standbyLeakageUwPerTile and
// steadyStateElapsedNsPerInference to be right -- it is their product
// (times 1e-3 for uW*ns -> pJ). Any one of the three left at its struct
// default makes this 0, which is why it is the single most load-bearing
// CHECK in this file: it cannot pass by accident.
// CHECK: "standby_leakage_pj_per_inference_floor": 1.08
