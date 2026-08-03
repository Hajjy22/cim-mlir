//===- erbium_backend.cpp - Real Erbium-8T hardware backend ----*- C++ -*-===//
//
// Stub. Real hardware bring-up is spec milestone M5 ("second target and
// generalization" / "community and real hardware"), which requires
// physical Erbium-8T access this repository cannot assume. Every entry
// point below fails with CIMRT_ERR_NO_DEVICE so that `cimrt_open` can
// distinguish "no real hardware attached" from "simulator unavailable"
// once dispatch-by-target-string exists (see runtime/CMakeLists.txt note).
//
//===----------------------------------------------------------------------===//

#include "cimrt.h"

namespace {
// TODO(spec M5): real Erbium-8T hardware bring-up. Until then this
// translation unit exists purely so a future dispatcher has a distinct,
// separately-linkable backend to route "erbium-8t-hw" (or similar) to,
// instead of the functional simulator.
cimrt_status erbium_not_yet_implemented() { return CIMRT_ERR_NO_DEVICE; }
} // namespace

namespace mlir {
namespace cim {
namespace erbium {

cimrt_status open() { return erbium_not_yet_implemented(); }
cimrt_status program() { return erbium_not_yet_implemented(); }
cimrt_status mvm() { return erbium_not_yet_implemented(); }

} // namespace erbium
} // namespace cim
} // namespace mlir
