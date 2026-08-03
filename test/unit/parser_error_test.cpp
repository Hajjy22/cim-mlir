//===- parser_error_test.cpp - Target parser rejection table ---*- C++ -*-===//
//
// The target reader handles a strict subset of YAML rather than delegating
// to a full implementation, so what it does NOT accept matters as much as
// what it does. A reader that silently skips a construct it does not
// understand drops configuration, and the resulting cost numbers look
// perfectly plausible.
//
// One table, one row per rejection branch, each asserting the diagnostic
// names the actual problem -- a parser that fails for the wrong reason is
// still a bug.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Target/TargetSpec.h"

#include <sstream>
#include <string>
#include <vector>

using namespace cim;

namespace {

/// A minimal well-formed document. Each negative case below mutates exactly
/// one thing, so a failure points at one branch.
std::string validDocument() {
  return R"(name: rejection-test
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

/// Replace the first occurrence of `from` with `to`.
std::string replaced(const std::string &from, const std::string &to) {
  std::string doc = validDocument();
  const size_t pos = doc.find(from);
  if (pos == std::string::npos)
    return doc; // the caller's CIM_EXPECT will catch this as a non-rejection
  return doc.replace(pos, from.size(), to);
}

std::string removed(const std::string &line) {
  std::string doc = validDocument();
  const size_t pos = doc.find(line);
  if (pos == std::string::npos)
    return doc;
  return doc.erase(pos, line.size());
}

struct Rejection {
  const char *name;
  std::string document;
  const char *expectedMessage;
};

bool parseText(const std::string &yaml, TargetSpec &spec, std::string *error) {
  std::istringstream in(yaml);
  return parseTargetSpec(in, spec, error);
}

} // namespace

CIM_TEST(parser_accepts_the_unmutated_document) {
  // The control: if the base document did not parse, every rejection below
  // could be passing for the wrong reason.
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(parseText(validDocument(), spec, &error));
  CIM_EXPECT_EQ(spec.tiles.count, 4u);
}

CIM_TEST(parser_rejects_every_malformed_document) {
  const std::vector<Rejection> cases = {
      // --- structural ---
      {"tab indentation", "name: x\ntiles:\n\tcount: 4\n", "tab"},
      {"sequence", "name: x\nsupported:\n  - a\n  - b\n", "sequence"},
      {"missing colon", "name: x\nthis line has no colon\n", "key: value"},
      {"empty key", "name: x\n: novalue\n", "empty key"},

      // --- missing required fields ---
      {"missing name", removed("name: rejection-test\n"), "'name'"},
      {"missing class", removed("class: near_memory\n"), "'class'"},
      {"missing tiles.count", removed("  count: 4\n"), "tiles.count"},
      {"missing tiles.rows", removed("  rows: 128\n"), "tiles.rows"},
      {"missing tiles.cols", removed("  cols: 128\n"), "tiles.cols"},
      {"missing weight_dtype", removed("  weight_dtype: i8\n"),
       "tiles.weight_dtype"},
      {"missing mvm energy", removed("    energy_pj: 20\n"),
       "costs.mvm.energy_pj"},
      {"missing effective bits", removed("  output_effective_bits: 8\n"),
       "precision.output_effective_bits"},

      // --- bad enum values ---
      {"unknown class", replaced("class: near_memory", "class: quantum_vibes"),
       "quantum_vibes"},
      {"unknown provenance",
       replaced("provenance: estimated", "provenance: vibes"), "vibes"},
      {"bad persistence",
       replaced("  persistence: nonvolatile", "  persistence: sometimes"),
       "persistence"},

      // --- bad scalars ---
      {"non-integer count", replaced("  count: 4", "  count: four"),
       "not an integer"},
      {"non-numeric latency",
       replaced("    latency_ns: 1000", "    latency_ns: fast"),
       "not a number"},
      {"non-boolean persistent",
       replaced("  persistent: true", "  persistent: maybe"),
       "not a boolean"},

      // --- semantically impossible devices ---
      // These parse cleanly but describe hardware that cannot exist.
      // Accepting them would produce confidently wrong cost numbers, which
      // is worse than a parse error.
      {"zero tiles", replaced("  count: 4", "  count: 0"), "tiles.count"},
      {"zero rows", replaced("  rows: 128", "  rows: 0"), "tiles.rows"},
      {"zero cols", replaced("  cols: 128", "  cols: 0"), "tiles.cols"},
      {"zero effective bits",
       replaced("  output_effective_bits: 8", "  output_effective_bits: 0"),
       "output_effective_bits"},
  };

  for (const Rejection &rejection : cases) {
    TargetSpec spec;
    std::string error;
    if (parseText(rejection.document, spec, &error)) {
      CIM_FAIL(std::string("parser ACCEPTED a malformed document: ") +
               rejection.name);
      continue;
    }
    if (error.empty()) {
      CIM_FAIL(std::string("parser rejected '") + rejection.name +
               "' with no diagnostic at all");
      continue;
    }
    CIM_EXPECT_CONTAINS(error, rejection.expectedMessage);
  }
}

CIM_TEST(parser_reports_a_missing_file_rather_than_defaulting) {
  TargetSpec spec;
  std::string error;
  CIM_EXPECT(
      !parseTargetSpecFromFile("targets/definitely-not-here.yaml", spec, &error));
  CIM_EXPECT_CONTAINS(error, "definitely-not-here");

  // The two-argument convenience overload logs to stderr; exercise it so it
  // is not the one untested entry point in the file.
  TargetSpec spec2;
  CIM_EXPECT(!parseTargetSpecFromFile("targets/definitely-not-here.yaml", spec2));
}

CIM_TEST(parser_accepts_the_documented_subset) {
  // The mirror image of the rejection table: things the format explicitly
  // supports must keep working, or a stricter parser would quietly break
  // every shipped target file.
  TargetSpec spec;
  std::string error;

  // Optional fields may be omitted entirely.
  std::string doc = removed("  standby_leakage_uw_per_tile: 0.0\n");
  doc = doc.substr(0, doc.find("capabilities:"));
  CIM_EXPECT(parseText(doc, spec, &error));
  CIM_EXPECT(!spec.capabilities.doubleBufferProgram); // default

  // Comments, blank lines and trailing inline comments.
  CIM_EXPECT(parseText("# leading\n\n" +
                           replaced("  count: 4", "  count: 4   # inline") +
                           "\n# trailing\n",
                       spec, &error));
  CIM_EXPECT_EQ(spec.tiles.count, 4u);

  // Quoted strings.
  CIM_EXPECT(parseText(replaced("name: rejection-test", "name: \"quoted\""),
                       spec, &error));
  CIM_EXPECT_EQ(spec.name, std::string("quoted"));
}
