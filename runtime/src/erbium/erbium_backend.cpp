//===- erbium_backend.cpp - Real Erbium-8T hardware backend ----*- C++ -*-===//
//
// Stub. Real hardware bring-up is spec milestone M5, which needs physical
// Erbium-8T access this repository cannot assume.
//
// This is not dead code, though: cimrt_open routes any target name ending
// in "-hw" here, so `cimrt_open("erbium-8t-hw", &d)` really does return
// CIMRT_ERR_NO_DEVICE, and the distinction cimrt.h promises between "no
// hardware attached" and "simulator unavailable" is one a caller can
// actually observe and a test can actually assert.
//
//===----------------------------------------------------------------------===//

#include "erbium_backend.h"

namespace cim {
namespace erbium {

namespace {
// TODO(spec M5): real Erbium-8T hardware bring-up. Every entry point
// reports the same truthful answer until then.
cimrt_status notAttached() { return CIMRT_ERR_NO_DEVICE; }
} // namespace

cimrt_status open(cimrt_device **out) {
  if (out)
    *out = nullptr;
  return notAttached();
}
cimrt_status program() { return notAttached(); }
cimrt_status mvm() { return notAttached(); }

} // namespace erbium
} // namespace cim
