// RUN: cim-opt %s "--cim-cost-report=target-yaml=%S/../targets/tiny-4x4.yaml output=%t.json"
// RUN: FileCheck --input-file=%t.json %s

// cim.program's own cost_ns/cost_pj attributes are dead IR: cim-partition
// sets them from the target file at emission time
// (lib/Transforms/CIMPartition.cpp), but nothing downstream ever reads
// them back. cim-cost-report (this test), cim-lower-to-target's
// lowerProgram, and the interpreter's runProgram each independently
// re-derive cost from a freshly parsed TargetSpec instead -- none of the
// three so much as calls op.getCostNs()/getCostPj() anywhere in this
// repository (confirmed by grep, not just by reading these three
// functions). CIMPartition.cpp's own comment on the line that sets them
// -- "cim.program carries its own cost so later passes can reason about
// reprogramming without a target lookup" -- states a design intent that
// is not what actually happens: every later pass performs exactly the
// target lookup that comment says is unnecessary.
//
// This hand-writes a cim.program whose cost_ns/cost_pj are 999999999 --
// a value that could never come from any real cim-partition run against
// tiny-4x4.yaml (whose costs.program declares latency_ns: 1000,
// energy_pj: 10000, see test/targets/tiny-4x4.yaml) -- and checks that
// cim-cost-report's own numbers are the TARGET's, not the IR's. If this
// pass, or a future one, ever started trusting the op's own attribute
// instead of the target file, this 999999999 would show up somewhere in
// the totals below and this test would catch it; today it appears
// nowhere in the output at all.
//
// No cim-detect/cim-partition/cim-placement in the RUN line above,
// deliberately: this op is not derived from a matmul the normal way (that
// would always emit the correct cost from the same target this test also
// passes to cim-cost-report, which would prove nothing about whether the
// IR's own copy is ever consulted). Hand-writing cim.program directly, the
// same way test/Transforms/cim-lower-to-target.mlir's own fixtures do, is
// what makes the IR's stated cost and the target's declared cost
// independently controllable.

func.func @straight_line() {
  %dev = cim.device_open {target = "t"} : !cim.device<"t">
  %t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"t">) -> !cim.tile<4x4xi8>
  %w = memref.get_global @w : memref<4x4xi8>
  %r = cim.program %t, %w {cost_ns = 999999999 : i64, cost_pj = 999999999 : i64}
       : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  %a = memref.alloc() : memref<4xi8>
  %out = cim.mvm %r, %a {accumulate = false}
       : (!cim.resident<4x4xi8>, memref<4xi8>) -> memref<4xi32>
  memref.dealloc %a : memref<4xi8>
  memref.dealloc %out : memref<4xi32>
  return
}
memref.global "private" constant @w : memref<4x4xi8> = dense<1>

// One program, one mvm, no loop: tiny-4x4.yaml's costs.program (latency_ns:
// 1000, energy_pj: 10000) and costs.mvm (latency_ns: 10, energy_pj: 100)
// are the only numbers that could produce these totals -- 999999999 is not
// within a factor of 10000 of either, so a regression that started reading
// the IR's own attribute could not accidentally still pass this check.
// CHECK: "programs": 1,
// CHECK: "mvms": 1,
// CHECK: "install_programs": 1,
// CHECK: "install_energy_pj": 10000,
// CHECK: "install_latency_ns": 1000,
// CHECK: "total_energy_pj": 10100,
// CHECK: "total_latency_ns": 1010,
// CHECK-NOT: 999999999
