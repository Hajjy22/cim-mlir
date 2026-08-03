//===- simulator.cpp - Functional simulator backend (spec Sec. 9.1) -*- C++ *-
//
// Pure integer arithmetic reference implementation of cimrt.h. Its only job
// is correctness (does the compiled artifact compute the same result as a
// reference torch.matmul?) — no cycle timing. Cost/energy accounting is
// delegated to CostAccumulator (cost_model.h), which sums the target file's
// declared costs analytically (spec Sec. 9.2), not simulated.
//
// `cimrt_open("erbium-8t", ...)` and any other target name both route here
// in the M0/M1 skeleton state (see runtime/src/erbium/ for the real,
// currently-stubbed, hardware backend).
//
//===----------------------------------------------------------------------===//

#include "cimrt.h"
#include "cim/Target/TargetSpec.h"
#include "cost_model.h"

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using namespace mlir::cim;

struct cimrt_device {
  std::string name;
  TargetSpec spec;
  CostAccumulator cost{spec};
  std::unordered_map<cimrt_tile_id, std::vector<uint8_t>> tileWeights;
  bool profiling = false;
};

struct cimrt_buffer {
  std::vector<uint8_t> data;
  cimrt_space space;
};

extern "C" {

cimrt_status cimrt_open(const char *target_name, cimrt_device **out) {
  if (!target_name || !out)
    return CIMRT_ERR_NO_DEVICE;

  auto *dev = new cimrt_device();
  dev->name = target_name;

  // TODO(spec Sec.7): v0.1 resolves target_name to targets/<name>.yaml
  // relative to the current working directory. A real target search path
  // (env var, install prefix, etc.) is future work.
  std::string path = std::string("targets/") + target_name + ".yaml";
  if (!parseTargetSpecFromFile(path, dev->spec)) {
    delete dev;
    return CIMRT_ERR_NO_DEVICE;
  }

  *out = dev;
  return CIMRT_OK;
}

void cimrt_close(cimrt_device *dev) { delete dev; }

cimrt_status cimrt_query(cimrt_device *dev, cimrt_device_info *out) {
  if (!dev || !out)
    return CIMRT_ERR_NO_DEVICE;
  std::memset(out, 0, sizeof(*out));
  std::strncpy(out->name, dev->name.c_str(), sizeof(out->name) - 1);
  out->num_tiles = dev->spec.tiles.count;
  out->tile_rows = dev->spec.tiles.rows;
  out->tile_cols = dev->spec.tiles.cols;
  out->persistent = dev->spec.tiles.persistent;
  return CIMRT_OK;
}

cimrt_status cimrt_alloc(cimrt_device *dev, size_t bytes, cimrt_space space,
                          cimrt_buffer **out) {
  if (!dev || !out)
    return CIMRT_ERR_NO_DEVICE;
  auto *buf = new cimrt_buffer();
  buf->data.resize(bytes, 0);
  buf->space = space;
  *out = buf;
  return CIMRT_OK;
}

cimrt_status cimrt_copy(cimrt_buffer *dst, const cimrt_buffer *src) {
  if (!dst || !src)
    return CIMRT_ERR_OOM;
  if (dst->data.size() != src->data.size())
    return CIMRT_ERR_SHAPE_MISMATCH;
  dst->data = src->data;
  return CIMRT_OK;
}

void cimrt_free(cimrt_buffer *buf) { delete buf; }

cimrt_status cimrt_program(cimrt_device *dev, cimrt_tile_id tile,
                            const cimrt_buffer *weights) {
  if (!dev || !weights)
    return CIMRT_ERR_NO_DEVICE;
  if (tile >= dev->spec.tiles.count)
    return CIMRT_ERR_TILE_BUSY;
  size_t expected =
      static_cast<size_t>(dev->spec.tiles.rows) * dev->spec.tiles.cols;
  if (weights->data.size() != expected)
    return CIMRT_ERR_SHAPE_MISMATCH;

  // Verifier rule 3 (spec Sec.5.4) at the runtime layer: re-programming a
  // tile simply replaces its resident weights; any !cim.resident SSA value
  // referring to the old contents is a compile-time-invalid use, which
  // cim-opt (not this runtime) is responsible for rejecting.
  dev->tileWeights[tile] = weights->data;

  dev->cost.recordProgram();
  return CIMRT_OK;
}

cimrt_status cimrt_mvm(cimrt_device *dev, cimrt_tile_id tile,
                        const cimrt_buffer *act, cimrt_buffer *out,
                        bool accumulate) {
  if (!dev || !act || !out)
    return CIMRT_ERR_NO_DEVICE;

  auto it = dev->tileWeights.find(tile);
  if (it == dev->tileWeights.end())
    return CIMRT_ERR_TILE_BUSY; // tile has never been cimrt_program'd

  const uint32_t rows = dev->spec.tiles.rows;
  const uint32_t cols = dev->spec.tiles.cols;
  const std::vector<uint8_t> &weights = it->second;

  // TODO(spec Sec.9.1): this reference implementation assumes i8 weights,
  // i8 activations, i32 accumulation (the v0.1 contract's INT8 matmul).
  // Other target-declared dtypes are unimplemented.
  if (act->data.size() != cols)
    return CIMRT_ERR_SHAPE_MISMATCH;
  if (out->data.size() != static_cast<size_t>(rows) * sizeof(int32_t))
    return CIMRT_ERR_SHAPE_MISMATCH;

  auto *outI32 = reinterpret_cast<int32_t *>(out->data.data());
  for (uint32_t r = 0; r < rows; ++r) {
    int32_t acc = accumulate ? outI32[r] : 0;
    for (uint32_t c = 0; c < cols; ++c) {
      auto w = static_cast<int8_t>(weights[static_cast<size_t>(r) * cols + c]);
      auto a = static_cast<int8_t>(act->data[c]);
      acc += static_cast<int32_t>(w) * static_cast<int32_t>(a);
    }
    outI32[r] = acc;
  }

  dev->cost.recordMvm();
  return CIMRT_OK;
}

cimrt_status cimrt_barrier(cimrt_device *dev) {
  if (!dev)
    return CIMRT_ERR_NO_DEVICE;
  // Functional simulator executes synchronously; nothing to wait on.
  return CIMRT_OK;
}

cimrt_status cimrt_profile_start(cimrt_device *dev) {
  if (!dev)
    return CIMRT_ERR_NO_DEVICE;
  dev->profiling = true;
  return CIMRT_OK;
}

cimrt_status cimrt_profile_stop(cimrt_device *dev, cimrt_profile *out) {
  if (!dev || !out)
    return CIMRT_ERR_NO_DEVICE;
  out->programs_issued = dev->cost.getProgramsIssued();
  out->mvms_issued = dev->cost.getMvmsIssued();
  out->bytes_transferred = dev->cost.getBytesTransferred();
  out->estimated_energy_pj = dev->cost.getEstimatedEnergyPj();
  out->estimated_latency_ns = dev->cost.getEstimatedLatencyNs();
  dev->profiling = false;
  return CIMRT_OK;
}

} // extern "C"
