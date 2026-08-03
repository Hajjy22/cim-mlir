//===- TargetSpec.h - Target description schema (spec Sec. 7) --*- C++ -*-===//
#ifndef CIM_TARGET_TARGETSPEC_H
#define CIM_TARGET_TARGETSPEC_H

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace cim {

/// One of near_memory | digital_cim | analog_cim | dpu (spec Sec. 7).
enum class TargetClass { NearMemory, DigitalCIM, AnalogCIM, DPU };

/// measured | simulated | estimated (spec Sec. 7 provenance discipline:
/// every number in a target file must carry a source).
enum class Provenance { Measured, Simulated, Estimated };

struct TileSpec {
  uint32_t count = 0;
  uint32_t rows = 0;
  uint32_t cols = 0;
  std::string weightDtype;      // e.g. "i8"
  std::string activationDtype;  // e.g. "i8"
  std::string accumulatorDtype; // e.g. "i32"
  bool persistent = false;
  std::string persistence; // "volatile" | "nonvolatile"
};

struct ProgramCost {
  double latencyNs = 0.0;
  double energyPj = 0.0;
};

struct MvmCost {
  double latencyNs = 0.0;
  double energyPj = 0.0;
};

struct TransferCost {
  double bandwidthGbps = 0.0;
  double energyPjPerByte = 0.0;
};

struct CostModel {
  ProgramCost program;
  MvmCost mvm;
  TransferCost transfer;
  double standbyLeakageUwPerTile = 0.0;
};

struct PrecisionSpec {
  uint32_t outputEffectiveBits = 8;
};

struct CapabilitiesSpec {
  bool doubleBufferProgram = false;
  bool partialSumInPlace = false;
  bool autonomousControl = false;
};

/// Mirrors the full YAML schema from spec Section 7 (see targets/*.yaml).
struct TargetSpec {
  std::string name;
  std::string description;
  std::string version;
  TargetClass targetClass = TargetClass::NearMemory;
  Provenance provenance = Provenance::Estimated;

  TileSpec tiles;
  CostModel costs;
  PrecisionSpec precision;
  CapabilitiesSpec capabilities;
};

/// TODO(spec Sec.7): implemented in TargetYAMLParser.cpp via
/// llvm::yaml::MappingTraits<TargetSpec>. Returns failure (and diagnostics
/// on stderr) on malformed YAML.
bool parseTargetSpecFromFile(const std::string &path, TargetSpec &outSpec);

} // namespace cim
} // namespace mlir

#endif // CIM_TARGET_TARGETSPEC_H
