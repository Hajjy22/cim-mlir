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
#include "../erbium/erbium_backend.h"

#include <cstring>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace cim;

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
  // Back-pointer to the owning device. Without it cimrt_copy has no device
  // handle, so CostAccumulator::recordTransfer was unreachable by
  // construction and bytes_transferred was permanently zero.
  cimrt_device *dev = nullptr;
};

extern "C" {

/// Resolve a target name to a file path. A name containing '/' or ending
/// in ".yaml" is taken as a path, so callers that are not run from the repo
/// root (the interpreter, tests) can still open a device.
static std::string resolveTargetPath(const std::string &name) {
  const bool looksLikePath =
      name.find('/') != std::string::npos ||
      (name.size() >= 5 && name.compare(name.size() - 5, 5, ".yaml") == 0);
  return looksLikePath ? name : "targets/" + name + ".yaml";
}

cimrt_status cimrt_open(const char *target_name, cimrt_device **out) {
  if (!target_name || !out)
    return CIMRT_ERR_INVALID_ARG;

  // A "-hw" suffix asks for real hardware, which is a different backend --
  // not this functional simulator. Reporting NO_DEVICE here is what makes
  // "no hardware attached" distinguishable from "simulator unavailable".
  const std::string name = target_name;
  if (name.size() >= 3 && name.compare(name.size() - 3, 3, "-hw") == 0)
    return cim::erbium::open();

  auto *dev = new cimrt_device();
  dev->name = name;

  if (!parseTargetSpecFromFile(resolveTargetPath(name), dev->spec)) {
    delete dev;
    return CIMRT_ERR_NO_DEVICE;
  }

  *out = dev;
  return CIMRT_OK;
}

void cimrt_close(cimrt_device *dev) { delete dev; }

cimrt_status cimrt_query(cimrt_device *dev, cimrt_device_info *out) {
  if (!dev || !out)
    return CIMRT_ERR_INVALID_ARG;
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
    return CIMRT_ERR_INVALID_ARG;
  auto *buf = new cimrt_buffer();
  buf->data.resize(bytes, 0);
  buf->space = space;
  buf->dev = dev;
  *out = buf;
  return CIMRT_OK;
}

cimrt_status cimrt_copy(cimrt_buffer *dst, const cimrt_buffer *src) {
  if (!dst || !src)
    return CIMRT_ERR_INVALID_ARG;
  if (dst->data.size() != src->data.size())
    return CIMRT_ERR_SHAPE_MISMATCH;
  dst->data = src->data;
  if (dst->dev)
    dst->dev->cost.recordTransfer(dst->data.size());
  return CIMRT_OK;
}

cimrt_status cimrt_write(cimrt_buffer *dst, size_t dst_offset, const void *src,
                          size_t bytes) {
  if (!dst || !src)
    return CIMRT_ERR_INVALID_ARG;
  // Checked against overflow rather than just `offset + bytes > size`,
  // which wraps for adversarial offsets.
  if (dst_offset > dst->data.size() || bytes > dst->data.size() - dst_offset)
    return CIMRT_ERR_INVALID_ARG;
  std::memcpy(dst->data.data() + dst_offset, src, bytes);
  if (dst->dev)
    dst->dev->cost.recordTransfer(bytes);
  return CIMRT_OK;
}

cimrt_status cimrt_read(const cimrt_buffer *src, size_t src_offset, void *dst,
                         size_t bytes) {
  if (!src || !dst)
    return CIMRT_ERR_INVALID_ARG;
  if (src_offset > src->data.size() || bytes > src->data.size() - src_offset)
    return CIMRT_ERR_INVALID_ARG;
  std::memcpy(dst, src->data.data() + src_offset, bytes);
  if (src->dev)
    src->dev->cost.recordTransfer(bytes);
  return CIMRT_OK;
}

void cimrt_free(cimrt_buffer *buf) { delete buf; }

cimrt_status cimrt_program(cimrt_device *dev, cimrt_tile_id tile,
                            const cimrt_buffer *weights) {
  if (!dev || !weights)
    return CIMRT_ERR_INVALID_ARG;
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
    return CIMRT_ERR_INVALID_ARG;

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

  // Accumulate in int64 and wrap explicitly on store. Accumulating straight
  // into an int32_t is signed-overflow UB as soon as K is large enough --
  // UBSan flags it, and the standard gives no guarantee about the result.
  // The documented behaviour of this model is to WRAP, matching a
  // fixed-width hardware accumulator, rather than to saturate.
  auto *outI32 = reinterpret_cast<int32_t *>(out->data.data());
  for (uint32_t r = 0; r < rows; ++r) {
    int64_t acc = accumulate ? static_cast<int64_t>(outI32[r]) : 0;
    for (uint32_t c = 0; c < cols; ++c) {
      auto w = static_cast<int8_t>(weights[static_cast<size_t>(r) * cols + c]);
      auto a = static_cast<int8_t>(act->data[c]);
      acc += static_cast<int64_t>(w) * static_cast<int64_t>(a);
    }
    outI32[r] = static_cast<int32_t>(static_cast<uint32_t>(acc));
  }

  dev->cost.recordMvm();
  return CIMRT_OK;
}

cimrt_status cimrt_barrier(cimrt_device *dev) {
  if (!dev)
    return CIMRT_ERR_INVALID_ARG;
  // Functional simulator executes synchronously; nothing to wait on.
  return CIMRT_OK;
}

cimrt_status cimrt_profile_start(cimrt_device *dev) {
  if (!dev)
    return CIMRT_ERR_INVALID_ARG;
  dev->profiling = true;
  return CIMRT_OK;
}

cimrt_status cimrt_profile_stop(cimrt_device *dev, cimrt_profile *out) {
  if (!dev || !out)
    return CIMRT_ERR_INVALID_ARG;
  out->programs_issued = dev->cost.getProgramsIssued();
  out->mvms_issued = dev->cost.getMvmsIssued();
  out->bytes_transferred = dev->cost.getBytesTransferred();
  out->estimated_energy_pj = dev->cost.getEstimatedEnergyPj();
  out->estimated_latency_ns = dev->cost.getEstimatedLatencyNs();
  dev->profiling = false;
  return CIMRT_OK;
}

const char *cimrt_status_string(cimrt_status status) {
  switch (status) {
  case CIMRT_OK: return "ok";
  case CIMRT_ERR_NO_DEVICE: return "no such device";
  case CIMRT_ERR_TILE_BUSY: return "tile unavailable or not programmed";
  case CIMRT_ERR_SHAPE_MISMATCH: return "buffer size does not match tile geometry";
  case CIMRT_ERR_OOM: return "out of memory";
  case CIMRT_ERR_INVALID_ARG: return "invalid argument";
  }
  return "unknown status";
}

} // extern "C"
