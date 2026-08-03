//===- TargetYAMLParser.cpp - Parse target/*.yaml (spec Sec. 7) -*- C++ -*-===//

#include "cim/Target/TargetSpec.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::cim;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::string)

namespace llvm {
namespace yaml {

template <> struct ScalarEnumerationTraits<TargetClass> {
  static void enumeration(IO &io, TargetClass &value) {
    io.enumCase(value, "near_memory", TargetClass::NearMemory);
    io.enumCase(value, "digital_cim", TargetClass::DigitalCIM);
    io.enumCase(value, "analog_cim", TargetClass::AnalogCIM);
    io.enumCase(value, "dpu", TargetClass::DPU);
  }
};

template <> struct ScalarEnumerationTraits<Provenance> {
  static void enumeration(IO &io, Provenance &value) {
    io.enumCase(value, "measured", Provenance::Measured);
    io.enumCase(value, "simulated", Provenance::Simulated);
    io.enumCase(value, "estimated", Provenance::Estimated);
  }
};

template <> struct MappingTraits<TileSpec> {
  static void mapping(IO &io, TileSpec &t) {
    io.mapRequired("count", t.count);
    io.mapRequired("rows", t.rows);
    io.mapRequired("cols", t.cols);
    io.mapRequired("weight_dtype", t.weightDtype);
    io.mapRequired("activation_dtype", t.activationDtype);
    io.mapOptional("accumulator_dtype", t.accumulatorDtype);
    io.mapOptional("persistent", t.persistent);
    io.mapOptional("persistence", t.persistence);
  }
};

template <> struct MappingTraits<ProgramCost> {
  static void mapping(IO &io, ProgramCost &c) {
    io.mapRequired("latency_ns", c.latencyNs);
    io.mapRequired("energy_pj", c.energyPj);
  }
};

template <> struct MappingTraits<MvmCost> {
  static void mapping(IO &io, MvmCost &c) {
    io.mapRequired("latency_ns", c.latencyNs);
    io.mapRequired("energy_pj", c.energyPj);
  }
};

template <> struct MappingTraits<TransferCost> {
  static void mapping(IO &io, TransferCost &c) {
    io.mapRequired("bandwidth_gbps", c.bandwidthGbps);
    io.mapRequired("energy_pj_per_byte", c.energyPjPerByte);
  }
};

template <> struct MappingTraits<CostModel> {
  static void mapping(IO &io, CostModel &c) {
    io.mapRequired("program", c.program);
    io.mapRequired("mvm", c.mvm);
    io.mapRequired("transfer", c.transfer);
    io.mapOptional("standby_leakage_uw_per_tile", c.standbyLeakageUwPerTile);
  }
};

template <> struct MappingTraits<PrecisionSpec> {
  static void mapping(IO &io, PrecisionSpec &p) {
    io.mapRequired("output_effective_bits", p.outputEffectiveBits);
  }
};

template <> struct MappingTraits<CapabilitiesSpec> {
  static void mapping(IO &io, CapabilitiesSpec &c) {
    io.mapOptional("double_buffer_program", c.doubleBufferProgram);
    io.mapOptional("partial_sum_in_place", c.partialSumInPlace);
    io.mapOptional("autonomous_control", c.autonomousControl);
  }
};

template <> struct MappingTraits<TargetSpec> {
  static void mapping(IO &io, TargetSpec &spec) {
    io.mapRequired("name", spec.name);
    io.mapOptional("description", spec.description);
    io.mapOptional("version", spec.version);
    io.mapRequired("class", spec.targetClass);
    io.mapOptional("provenance", spec.provenance);
    io.mapRequired("tiles", spec.tiles);
    io.mapRequired("costs", spec.costs);
    io.mapRequired("precision", spec.precision);
    io.mapOptional("capabilities", spec.capabilities);
  }
};

} // namespace yaml
} // namespace llvm

bool mlir::cim::parseTargetSpecFromFile(const std::string &path,
                                         TargetSpec &outSpec) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    llvm::errs() << "cim: failed to open target file '" << path
                 << "': " << bufferOrErr.getError().message() << "\n";
    return false;
  }

  llvm::yaml::Input yin(bufferOrErr.get()->getBuffer());
  yin >> outSpec;
  if (yin.error()) {
    llvm::errs() << "cim: failed to parse target file '" << path
                 << "': " << yin.error().message() << "\n";
    return false;
  }
  return true;
}
