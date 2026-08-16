//===- workload_json_test.cpp - --emit-workload JSON reader table -*- C++-*-===//
//
// Same discipline as parser_error_test.cpp's target-YAML table: one
// control case proving the base document parses, then one row per
// rejection branch, each asserting the diagnostic names the actual
// problem. A hand-rolled reader that fails for the wrong reason is still
// a bug, and this schema's own honesty requirement (WorkloadDocument's
// `skipped` field) makes "missing 'skipped' is refused, not defaulted to
// empty" a case worth pinning directly rather than trusting to inspection.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cim/Placement/WorkloadJSON.h"

using namespace cim;

namespace {

std::string validDocument() {
  return R"({
  "model": "m.onnx",
  "layers": [
    {"name": "conv1", "op_type": "QLinearConv", "k": 27, "n": 8},
    {"name": "mm1", "op_type": "MatMulInteger", "k": 8, "n": 4}
  ],
  "skipped": [
    {"name": "pool1", "op_type": "MaxPool", "reason": "no resident weights"}
  ]
})";
}

bool parse(const std::string &text, WorkloadDocument &doc,
          std::string *error) {
  return parseWorkloadDocument(text, doc, error);
}

} // namespace

CIM_TEST(workload_json_accepts_the_unmutated_document) {
  // The control: if this does not parse, every rejection case below could
  // be passing for the wrong reason.
  WorkloadDocument doc;
  std::string error;
  CIM_EXPECT(parse(validDocument(), doc, &error));
  CIM_EXPECT_EQ(doc.model, "m.onnx");
  CIM_EXPECT_EQ(doc.layers.size(), 2u);
  CIM_EXPECT_EQ(doc.layers[0].name, "conv1");
  CIM_EXPECT_EQ(doc.layers[0].opType, "QLinearConv");
  CIM_EXPECT_EQ(doc.layers[0].k, 27u);
  CIM_EXPECT_EQ(doc.layers[0].n, 8u);
  CIM_EXPECT_EQ(doc.layers[1].k, 8u);
  CIM_EXPECT_EQ(doc.layers[1].n, 4u);
  CIM_EXPECT_EQ(doc.skipped.size(), 1u);
  CIM_EXPECT_EQ(doc.skipped[0].opType, "MaxPool");
  CIM_EXPECT_EQ(doc.skipped[0].reason, "no resident weights");
}

CIM_TEST(workload_json_empty_layers_and_skipped_are_valid) {
  WorkloadDocument doc;
  std::string error;
  CIM_EXPECT(
      parse(R"({"model": "m", "layers": [], "skipped": []})", doc, &error));
  CIM_EXPECT(doc.layers.empty());
  CIM_EXPECT(doc.skipped.empty());
}

CIM_TEST(workload_json_missing_skipped_is_refused_not_defaulted) {
  // The honesty requirement, pinned: a document that never mentions
  // `skipped` must not be silently read as "nothing was skipped".
  WorkloadDocument doc;
  std::string error;
  CIM_EXPECT(!parse(R"({"model": "m", "layers": []})", doc, &error));
  CIM_EXPECT_CONTAINS(error, "skipped");
}

CIM_TEST(workload_json_rejects_every_malformed_document) {
  struct Case {
    const char *name;
    std::string text;
    const char *expectedFragment;
  };
  const std::vector<Case> cases = {
      {"missing model", R"({"layers": [], "skipped": []})", "model"},
      {"missing layers", R"({"model": "m", "skipped": []})", "layers"},
      {"layers not an array", R"({"model": "m", "layers": {}, "skipped": []})",
       "must be an array"},
      {"skipped not an array",
       R"({"model": "m", "layers": [], "skipped": {}})", "must be an array"},
      {"layer missing name",
       R"({"model": "m", "layers": [{"op_type": "MatMulInteger", "k": 1, "n": 1}], "skipped": []})",
       "'name'"},
      {"layer missing k",
       R"({"model": "m", "layers": [{"name": "a", "op_type": "MatMulInteger", "n": 1}], "skipped": []})",
       "'k'"},
      {"negative k",
       R"({"model": "m", "layers": [{"name": "a", "op_type": "MatMulInteger", "k": -1, "n": 1}], "skipped": []})",
       "non-negative integer"},
      {"fractional n",
       R"({"model": "m", "layers": [{"name": "a", "op_type": "MatMulInteger", "k": 1, "n": 4.5}], "skipped": []})",
       "non-negative integer"},
      {"k not a number",
       R"({"model": "m", "layers": [{"name": "a", "op_type": "MatMulInteger", "k": "8", "n": 1}], "skipped": []})",
       "non-negative integer"},
      {"skip entry missing reason",
       R"({"model": "m", "layers": [], "skipped": [{"name": "a", "op_type": "Relu"}]})",
       "'reason'"},
      {"zero k",
       R"({"model": "m", "layers": [{"name": "a", "op_type": "MatMulInteger", "k": 0, "n": 4}], "skipped": []})",
       "positive"},
      {"zero n",
       R"({"model": "m", "layers": [{"name": "a", "op_type": "MatMulInteger", "k": 4, "n": 0}], "skipped": []})",
       "positive"},
      {"top level is an array", R"([1, 2, 3])", "object"},
      {"truncated JSON", R"({"model": "m", "layers": [)", ""},
      {"trailing garbage",
       R"({"model": "m", "layers": [], "skipped": []} garbage)",
       "trailing content"},
      {"unterminated string", R"({"model": "m)", "unterminated"},
      {"single-quoted string", R"({'model': 'm'})", "string key"},
      {"trailing comma",
       R"({"model": "m", "layers": [], "skipped": [],})", ""},
  };

  for (const Case &c : cases) {
    WorkloadDocument doc;
    std::string error;
    const bool ok = parse(c.text, doc, &error);
    if (ok) {
      ::cimtest::reportFailure(__FILE__, __LINE__,
                               std::string(c.name) +
                                   ": expected a refusal, but it parsed");
      continue;
    }
    if (c.expectedFragment[0] != '\0')
      CIM_EXPECT_CONTAINS(error, c.expectedFragment);
  }
}

CIM_TEST(workload_json_decodes_unicode_escapes_as_utf8) {
  WorkloadDocument doc;
  std::string error;
  CIM_EXPECT(parse(
      R"({"model": "m", "layers": [{"name": "café", "op_type": "MatMulInteger", "k": 1, "n": 1}], "skipped": []})",
      doc, &error));
  CIM_EXPECT_EQ(doc.layers[0].name, "caf\xc3\xa9");
}

CIM_TEST(workload_json_from_file_names_the_path_on_failure) {
  WorkloadDocument doc;
  std::string error;
  CIM_EXPECT(!parseWorkloadDocumentFromFile(
      "/nonexistent/path/does-not-exist.json", doc, &error));
  CIM_EXPECT_CONTAINS(error, "does-not-exist.json");
}

CIM_TEST(workload_json_uint32_boundary_is_accepted_one_past_is_refused) {
  WorkloadDocument doc;
  std::string error;
  CIM_EXPECT(parse(
      R"({"model": "m", "layers": [{"name": "a", "op_type": "MatMulInteger", "k": 4294967295, "n": 1}], "skipped": []})",
      doc, &error));
  CIM_EXPECT_EQ(doc.layers[0].k, 4294967295u);

  WorkloadDocument doc2;
  CIM_EXPECT(!parse(
      R"({"model": "m", "layers": [{"name": "a", "op_type": "MatMulInteger", "k": 4294967296, "n": 1}], "skipped": []})",
      doc2, &error));
}
