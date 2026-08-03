// RUN: cim-opt %s | cim-opt | FileCheck %s

// CHECK-LABEL: func.func @open_device
func.func @open_device() -> !cim.device<"erbium-8t"> {
  // CHECK: cim.device_open
  %dev = cim.device_open {target = "erbium-8t"} : !cim.device<"erbium-8t">
  return %dev : !cim.device<"erbium-8t">
}
