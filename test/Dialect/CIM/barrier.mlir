// RUN: cim-opt %s | cim-opt | FileCheck %s

// CHECK-LABEL: func.func @barrier
func.func @barrier(%dev: !cim.device<"erbium-8t">) {
  // CHECK: cim.barrier
  cim.barrier %dev : !cim.device<"erbium-8t">
  return
}
