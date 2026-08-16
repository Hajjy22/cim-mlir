//===- WorkloadJSON.h - Read a cim-import-onnx --emit-workload file --*- C++-*-===//
//
// Reads the JSON `cim-import-onnx --emit-workload` produces
// (python/cim_frontend/analyze.py): a permissive graph walk's offloadable-
// layer shapes, for driving cim-bench's placement/cost engine against a
// real model without ever compiling or executing it. Weight residency
// pressure is a function of shape alone -- see analyze.py's own "THE
// INSIGHT THIS MODULE ACTS ON" note -- so this is the entire interchange
// this project's placement engine needs from a real network.
//
// A small, dependency-free JSON reader, same rationale and the same
// tradeoff as lib/Target/TargetYAMLParser.cpp's YAML reader: cim-bench and
// cimPlacement build with no LLVM/MLIR toolchain (the core, core-asan and
// core-valgrind CI jobs configure `-DCIM_ENABLE_MLIR=OFF`), so a JSON
// library is not available here either, and a hand-rolled reader accepts
// only what the schema below needs -- objects, arrays, strings, and
// numbers -- rather than being a general-purpose JSON implementation.
// Malformed input is refused with a message naming what was expected and
// where, matching TargetYAMLParser's own discipline, rather than guessed
// at or silently defaulted.
//
// The schema (also documented in analyze.py's own docstring):
//
//   {"model": "<path>",
//    "layers":  [{"name": "...", "op_type": "...", "k": <uint>, "n": <uint>}, ...],
//    "skipped": [{"name": "...", "op_type": "...", "reason": "..."}, ...]}
//
// `k`/`n` match partitionBlockCount's own convention (k = contraction
// dimension, n = output-channel dimension) -- Workloads.h.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_PLACEMENT_WORKLOADJSON_H
#define CIM_PLACEMENT_WORKLOADJSON_H

#include <cstdint>
#include <string>
#include <vector>

namespace cim {

/// One offloadable layer: a matmul or convolution whose weight shape was
/// known without needing to execute anything.
struct WorkloadLayer {
  std::string name;
  std::string opType;
  /// Both required to be strictly positive by the reader (WorkloadJSON.cpp)
  /// -- a matmul with a zero contraction dimension or zero output channels
  /// is not a degenerate real layer, it is not a layer at all, and
  /// partitionBlockCount would silently place zero blocks for it while
  /// this layer still counted toward "layers_analyzed".
  uint32_t k = 0;
  uint32_t n = 0;
};

/// One op the analysis walker declined to make resident, and why -- kept
/// alongside `layers` so a consumer can never see the offloadable count
/// without also seeing what was left out.
struct WorkloadSkip {
  std::string name;
  std::string opType;
  std::string reason;
};

/// The parsed document. `layers` and `skipped` are BOTH required fields in
/// the source JSON (each may be an empty array, but the key must be
/// present) -- a document missing `skipped` entirely cannot be
/// distinguished from one that omitted it by mistake, so it is refused
/// rather than silently treated as "nothing was skipped".
struct WorkloadDocument {
  std::string model;
  std::vector<WorkloadLayer> layers;
  std::vector<WorkloadSkip> skipped;
};

/// Parses `path` as a --emit-workload document. Returns false and fills
/// `*error` (if non-null) on any structural problem: the file could not be
/// opened, is not valid JSON, is missing a required field, or has a field
/// of the wrong type (including a negative or non-integer `k`/`n`).
bool parseWorkloadDocumentFromFile(const std::string &path,
                                    WorkloadDocument &doc, std::string *error);

/// Same, from an in-memory string -- what the file-based overload and its
/// differential test (test/python/test_workload_json_differential.py, run
/// against test/unit/workload_json_test.cpp's own exposed cases) both use.
bool parseWorkloadDocument(const std::string &text, WorkloadDocument &doc,
                           std::string *error);

} // namespace cim

#endif // CIM_PLACEMENT_WORKLOADJSON_H
