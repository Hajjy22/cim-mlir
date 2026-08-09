// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-cost-report=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file 2>&1 1>/dev/null | FileCheck --check-prefix=DIAG %s
// RUN: cim-opt %s --cim-detect \
// RUN:   --cim-partition=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --cim-cost-report=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file 2>/dev/null | FileCheck --check-prefix=JSON %s

// cim-run --profile checks the runtime numbers actually match
// (test/mlir/cost_report_e2e_test.cpp is the differential); this file
// checks the shape of what cim-cost-report prints, and the one path a
// runtime differential cannot reach: a trip count that is genuinely not a
// compile-time constant.

// A single 4x4 matmul, no loop -- the baseline JSON shape every field name
// below must appear with, matching the schema toJson() already pins in
// test/unit/cost_report_test.cpp (one format, not a second one for IR).

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

// JSON: trip-count-complete: true
// JSON: "label":
// JSON: "persistent": true,
// JSON: "programs": 1,
// JSON: "mvms": 1,
// JSON: "requantizes": 0,
// JSON: "reduce_partial_adds": 0,
// JSON: "reuses": 0,
// JSON: "install_programs": 1,
// JSON: "install_energy_pj":
// JSON: "install_latency_ns":
// JSON: "steady_state_programs_per_inference": 0,
// JSON: "total_energy_pj":
// JSON: "total_latency_ns":

// -----

// A loop whose trip count is a function argument -- genuinely not a
// compile-time constant, not just one this pass declines to prove. Its
// cim.program/cim.mvm sites must be excluded from the totals rather than
// assumed to fire once, and the pass must say so both in the JSON header
// and as a diagnostic.

memref.global "private" constant @w2 : memref<4x4xi8> = dense<1>
memref.global "private" constant @a2 : memref<1x4xi8> = dense<2>

func.func private @sink2(memref<1x4xi32>)

func.func @dynamic_bound(%n: index) {
  %w = memref.get_global @w2 : memref<4x4xi8>
  %i0 = memref.get_global @a2 : memref<1x4xi8>
  %a = memref.alloc() : memref<1x4xi8>
  memref.copy %i0, %a : memref<1x4xi8> to memref<1x4xi8>
  %out = memref.alloc() : memref<1x4xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    linalg.matmul_transpose_b ins(%a, %w : memref<1x4xi8>, memref<4x4xi8>)
      outs(%out : memref<1x4xi32>)
  }
  func.call @sink2(%out) : (memref<1x4xi32>) -> ()
  return
}

// JSON: trip-count-complete: false (unknown_program_sites=1, unknown_mvm_sites=1, unknown_requantize_sites=0, unknown_reduce_sites=0)
// JSON: "programs": 0,
// JSON: "mvms": 0,

// DIAG: cim-cost-report: 1 cim.program site(s), 1 cim.mvm site(s), 0 cim.requantize site(s), and 0 cim.reduce_partial site(s) sit under a loop whose trip count is not a compile-time constant
