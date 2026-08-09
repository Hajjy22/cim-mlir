//===- target_spec_test.cpp - Tests for the target file reader -*- C++ -*-===//
//
// The reader in lib/Target/TargetYAMLParser.cpp handles a strict subset of
// YAML rather than delegating to a full implementation, so it earns that
// choice here: every shipped target file is parsed and checked field by
// field, and the malformed cases are checked to fail rather than silently
// producing a plausible-looking device.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Target/TargetSpec.h"

#include <sstream>
#include <string>

using namespace cim;

namespace {

/// Locates targets/ relative to the source tree. The test binary may be run
/// from the build directory or the repo root, so try both.
std::string targetPath(const std::string &name) {
  static const char *prefixes[] = {"targets/", "../targets/", "../../targets/"};
  for (const char *prefix : prefixes) {
    std::string candidate = std::string(prefix) + name;
    TargetSpec probe;
    std::string ignored;
    if (parseTargetSpecFromFile(candidate, probe, &ignored))
      return candidate;
  }
  return "targets/" + name;
}

TargetSpec parseOrFail(const std::string &name) {
  TargetSpec spec;
  std::string error;
  if (!parseTargetSpecFromFile(targetPath(name), spec, &error))
    ::cimtest::reportFailure(__FILE__, __LINE__,
                             "failed to parse " + name + ": " + error);
  return spec;
}

bool parseText(const std::string &yaml, TargetSpec &spec, std::string *error) {
  std::istringstream in(yaml);
  return parseTargetSpec(in, spec, error);
}

/// A minimal well-formed document, used as the base for negative tests so
/// each one varies exactly the thing under test.
std::string validDocument() {
  return R"(name: unit-test-target
class: near_memory
provenance: estimated
tiles:
  count: 4
  rows: 128
  cols: 128
  weight_dtype: i8
  activation_dtype: i8
  accumulator_dtype: i32
  persistent: true
  persistence: nonvolatile
costs:
  program:
    latency_ns: 1000
    energy_pj: 2000
  mvm:
    latency_ns: 10
    energy_pj: 20
  requantize:
    latency_ns: 5
    energy_pj: 10
  reduce_partial:
    latency_ns: 3
    energy_pj: 6
  transfer:
    bandwidth_gbps: 8.0
    energy_pj_per_byte: 1.5
  standby_leakage_uw_per_tile: 0.0
precision:
  output_effective_bits: 8
capabilities:
  double_buffer_program: false
  partial_sum_in_place: true
  autonomous_control: true
)";
}

} // namespace

//===----------------------------------------------------------------------===//
// The shipped target files
//===----------------------------------------------------------------------===//

CIM_TEST(erbium_8t_parses_with_the_documented_values) {
  TargetSpec spec = parseOrFail("erbium-8t.yaml");

  CIM_EXPECT_EQ(spec.name, std::string("erbium-8t"));
  CIM_EXPECT(spec.targetClass == TargetClass::NearMemory);
  // v0.1 Erbium numbers are estimates, and the file must say so — a cost
  // model built on numbers claiming to be measured when they are not is
  // the failure mode spec Sec. 7 warns about.
  CIM_EXPECT(spec.provenance == Provenance::Estimated);

  CIM_EXPECT_EQ(spec.tiles.count, 8u);
  CIM_EXPECT_EQ(spec.tiles.rows, 256u);
  CIM_EXPECT_EQ(spec.tiles.cols, 256u);
  CIM_EXPECT_EQ(spec.tiles.weightDtype, std::string("i8"));
  CIM_EXPECT_EQ(spec.tiles.accumulatorDtype, std::string("i32"));
  CIM_EXPECT(spec.tiles.persistent);
  CIM_EXPECT_EQ(spec.tiles.persistence, std::string("nonvolatile"));

  CIM_EXPECT(spec.costs.program.latencyNs == 12000.0);
  CIM_EXPECT(spec.costs.program.energyPj == 480000.0);
  CIM_EXPECT(spec.costs.mvm.latencyNs == 640.0);
  CIM_EXPECT(spec.costs.mvm.energyPj == 3100.0);
  CIM_EXPECT(spec.costs.requantize.latencyNs == 50.0);
  CIM_EXPECT(spec.costs.requantize.energyPj == 150.0);
  CIM_EXPECT(spec.costs.transfer.bandwidthGbps == 12.8);
  CIM_EXPECT(spec.costs.transfer.energyPjPerByte == 4.2);
  // The non-volatile advantage: zero standby leakage.
  CIM_EXPECT(spec.costs.standbyLeakageUwPerTile == 0.0);

  CIM_EXPECT_EQ(spec.precision.outputEffectiveBits, 8u);
  CIM_EXPECT(!spec.capabilities.doubleBufferProgram);
  CIM_EXPECT(spec.capabilities.partialSumInPlace);
  CIM_EXPECT(spec.capabilities.autonomousControl);
}

CIM_TEST(generic_digital_cim_parses_as_a_different_class) {
  TargetSpec spec = parseOrFail("generic-digital-cim.yaml");
  CIM_EXPECT_EQ(spec.name, std::string("generic-digital-cim"));
  // Retargetability is the point (spec M4): a second target of a different
  // class must round-trip through the same schema.
  CIM_EXPECT(spec.targetClass == TargetClass::DigitalCIM);
  CIM_EXPECT(spec.provenance == Provenance::Estimated);
  CIM_EXPECT(!spec.tiles.persistent);
  CIM_EXPECT_EQ(spec.tiles.persistence, std::string("volatile"));
  // Volatile SRAM leaks; that must not be modeled as free.
  CIM_EXPECT(spec.costs.standbyLeakageUwPerTile > 0.0);
  // Reloading SRAM is far cheaper than a nonvolatile write.
  CIM_EXPECT(spec.costs.program.energyPj < 480000.0);
}

CIM_TEST(upmem_like_parses_as_dpu_class) {
  TargetSpec spec = parseOrFail("upmem-like.yaml");
  CIM_EXPECT_EQ(spec.name, std::string("upmem-like"));
  CIM_EXPECT(spec.targetClass == TargetClass::DPU);
  CIM_EXPECT(spec.provenance == Provenance::Estimated);
  CIM_EXPECT(spec.capabilities.autonomousControl);
}

CIM_TEST(every_shipped_target_declares_its_provenance) {
  // Spec Sec. 7 provenance discipline: no target file may leave the source
  // of its numbers unstated. All three shipped files are estimates today.
  for (const char *name :
       {"erbium-8t.yaml", "generic-digital-cim.yaml", "upmem-like.yaml"}) {
    TargetSpec spec = parseOrFail(name);
    CIM_EXPECT(spec.provenance == Provenance::Estimated);
  }
}

//===----------------------------------------------------------------------===//
// Parser mechanics
//===----------------------------------------------------------------------===//

CIM_TEST(valid_document_round_trips) {
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(parseText(validDocument(), spec, &error));
  CIM_EXPECT_EQ(spec.tiles.count, 4u);
  CIM_EXPECT(spec.costs.program.energyPj == 2000.0);
  CIM_EXPECT(spec.costs.transfer.bandwidthGbps == 8.0);
}

CIM_TEST(comments_and_blank_lines_are_ignored) {
  std::string doc = validDocument();
  doc = "# leading comment\n\n" + doc + "\n# trailing comment\n";
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(parseText(doc, spec, &error));
  CIM_EXPECT_EQ(spec.name, std::string("unit-test-target"));
}

CIM_TEST(trailing_comments_do_not_corrupt_values) {
  // The shipped files carry inline comments on numeric lines, so this is a
  // real path, not a hypothetical one.
  std::string doc = validDocument();
  const std::string from = "  count: 4";
  const std::string to = "  count: 4        # eight would be nicer";
  doc.replace(doc.find(from), from.size(), to);
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(parseText(doc, spec, &error));
  CIM_EXPECT_EQ(spec.tiles.count, 4u);
}

CIM_TEST(quoted_strings_are_unquoted) {
  std::string doc = validDocument();
  const std::string from = "name: unit-test-target";
  const std::string to = "name: \"quoted-target\"";
  doc.replace(doc.find(from), from.size(), to);
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(parseText(doc, spec, &error));
  CIM_EXPECT_EQ(spec.name, std::string("quoted-target"));
}

//===----------------------------------------------------------------------===//
// Rejection: a bad target file must fail loudly, not produce a wrong device
//===----------------------------------------------------------------------===//

CIM_TEST(missing_required_field_is_rejected) {
  std::string doc = validDocument();
  const std::string from = "  rows: 128\n";
  doc.erase(doc.find(from), from.size());
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(!parseText(doc, spec, &error));
  CIM_EXPECT(error.find("tiles.rows") != std::string::npos);
}

CIM_TEST(missing_requantize_cost_is_rejected) {
  // costs.requantize is required, the same as program/mvm/transfer: leaving
  // it unset would silently cost every cim.requantize at zero rather than
  // say so, which is exactly the "confidently wrong number" this parser's
  // required-field discipline exists to prevent.
  std::string doc = validDocument();
  const std::string from = "  requantize:\n    latency_ns: 5\n    energy_pj: 10\n";
  doc.erase(doc.find(from), from.size());
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(!parseText(doc, spec, &error));
  CIM_EXPECT(error.find("costs.requantize") != std::string::npos);
}

CIM_TEST(missing_reduce_partial_cost_is_rejected) {
  // costs.reduce_partial is required, the same as requantize: an N-operand
  // cim.reduce_partial lowers to N-1 real cimrt_reduce_add calls
  // (lowerReducePartial, lib/Transforms/CIMLowerToTarget.cpp), so leaving
  // this unset would silently cost every one of those calls at zero.
  std::string doc = validDocument();
  const std::string from =
      "  reduce_partial:\n    latency_ns: 3\n    energy_pj: 6\n";
  doc.erase(doc.find(from), from.size());
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(!parseText(doc, spec, &error));
  CIM_EXPECT(error.find("costs.reduce_partial") != std::string::npos);
}

CIM_TEST(unknown_target_class_is_rejected) {
  std::string doc = validDocument();
  const std::string from = "class: near_memory";
  const std::string to = "class: quantum_vibes";
  doc.replace(doc.find(from), from.size(), to);
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(!parseText(doc, spec, &error));
  CIM_EXPECT(error.find("quantum_vibes") != std::string::npos);
}

CIM_TEST(non_numeric_cost_is_rejected) {
  std::string doc = validDocument();
  const std::string from = "    latency_ns: 1000";
  const std::string to = "    latency_ns: fast";
  doc.replace(doc.find(from), from.size(), to);
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(!parseText(doc, spec, &error));
}

CIM_TEST(zero_tile_count_is_rejected) {
  std::string doc = validDocument();
  const std::string from = "  count: 4";
  const std::string to = "  count: 0";
  doc.replace(doc.find(from), from.size(), to);
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(!parseText(doc, spec, &error));
}

CIM_TEST(bad_persistence_value_is_rejected) {
  std::string doc = validDocument();
  const std::string from = "  persistence: nonvolatile";
  const std::string to = "  persistence: sometimes";
  doc.replace(doc.find(from), from.size(), to);
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(!parseText(doc, spec, &error));
}

CIM_TEST(unsupported_yaml_constructs_are_rejected_not_misread) {
  // The reader handles a subset. What it does not handle it must reject,
  // because silently skipping a sequence would drop configuration.
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(!parseText("name: x\nsupported:\n  - a\n  - b\n", spec, &error));

  std::string tabbed = "name: x\ntiles:\n\tcount: 4\n";
  CIM_EXPECT(!parseText(tabbed, spec, &error));
}

CIM_TEST(missing_file_is_reported_not_silently_defaulted) {
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(!parseTargetSpecFromFile("targets/does-not-exist.yaml", spec, &error));
  CIM_EXPECT(error.find("does-not-exist") != std::string::npos);
}
