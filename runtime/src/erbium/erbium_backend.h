//===- erbium_backend.h - Real Erbium hardware backend ---------*- C++ -*-===//
//
// Declares the hardware backend's entry points. Previously these were
// defined in erbium_backend.cpp but declared in no header at all, which
// made them unreachable dead code -- nothing could call them, so the
// "no hardware attached" path the cimrt.h contract promises did not exist.
//
// Namespace is `cim::erbium`, not `mlir::cim::erbium`: this translation
// unit has no MLIR dependency and does not belong under mlir::.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_RUNTIME_ERBIUM_BACKEND_H
#define CIM_RUNTIME_ERBIUM_BACKEND_H

#include "cimrt.h"

namespace cim {
namespace erbium {

/// Open real Erbium hardware. Returns CIMRT_ERR_NO_DEVICE until hardware
/// bring-up (spec M5); that is a truthful answer, not a placeholder --
/// there is genuinely no device attached.
cimrt_status open();

/// TODO(spec M5): real programming and compute against attached hardware.
cimrt_status program();
cimrt_status mvm();

} // namespace erbium
} // namespace cim

#endif // CIM_RUNTIME_ERBIUM_BACKEND_H
