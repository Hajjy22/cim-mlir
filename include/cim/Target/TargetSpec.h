//===- TargetSpec.h - Target description schema (spec Sec. 7) --*- C++ -*-===//
#ifndef CIM_TARGET_TARGETSPEC_H
#define CIM_TARGET_TARGETSPEC_H

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

// Plain `cim`, not `mlir::cim`: this schema and the placement/cost engines
// built on it carry no MLIR dependency, so they stay outside the mlir
// namespace. Only the dialect and passes live in `mlir::cim`.
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
  /// The field every pass that branches on volatility actually reads
  /// (cim-partition writes it straight onto cim.program's own `persistent`
  /// attribute; lib/Placement/CostReport.cpp's amortization curve shape
  /// depends on it).
  bool persistent = false;
  /// "volatile" | "nonvolatile", optional. Documentation-only: no pass
  /// reads this back -- `persistent` above is what actually drives cost-
  /// model behavior. Kept only because it reads better in a target file
  /// than a bare bool, and cross-checked against `persistent` by the
  /// parser (TargetYAMLParser.cpp) when both are present, so the two
  /// cannot silently describe two different devices.
  std::string persistence;
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

// A cim.requantize's cost, same fixed-per-op shape as program/mvm (unlike
// transfer, which scales with payload size). Modeled separately from mvm
// because it is a distinct hardware step -- an ADC readout on an analog
// target, a narrowing/rounding step on a digital one (spec Sec. 5.3) --
// not a free byproduct of the multiply-accumulate itself.
struct RequantizeCost {
  double latencyNs = 0.0;
  double energyPj = 0.0;
};

// One cim.reduce_partial chained add's cost -- same fixed-per-call shape as
// requantize, charged once per cimrt_reduce_add call an N-operand
// cim.reduce_partial lowers to (N-1 calls, not one per op site; see
// CostReportUtils.cpp). Modeled separately from mvm because it is a
// distinct step -- summing two already-computed partial accumulators, not
// part of the multiply-accumulate that produced either of them.
struct ReducePartialCost {
  double latencyNs = 0.0;
  double energyPj = 0.0;
};

struct CostModel {
  ProgramCost program;
  MvmCost mvm;
  TransferCost transfer;
  RequantizeCost requantize;
  ReducePartialCost reducePartial;
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

/// Parse a target description (spec Sec. 7, docs/target-format.md).
/// Rejects — rather than guesses at — missing required fields, unknown
/// enum values, and geometries that cannot describe a real device.
///
/// The `error` overloads report what was wrong; the two-argument form logs
/// to stderr and is the convenience entry point for tools.
bool parseTargetSpec(std::istream &in, TargetSpec &outSpec, std::string *error);
bool parseTargetSpecFromFile(const std::string &path, TargetSpec &outSpec,
                              std::string *error);
bool parseTargetSpecFromFile(const std::string &path, TargetSpec &outSpec);

} // namespace cim

#endif // CIM_TARGET_TARGETSPEC_H
