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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace cim;

struct cimrt_device {
  std::string name;
  TargetSpec spec;
  CostAccumulator cost{spec};
  std::unordered_map<cimrt_tile_id, std::vector<uint8_t>> tileWeights;
  // Snapshot of `cost`'s counters taken by cimrt_profile_start, so stop can
  // report the delta for one window rather than the cumulative total since
  // cimrt_open. Un-set (nullopt-by-flag) until start is actually called.
  CostAccumulator::Snapshot profileBaseline{};
  bool profilingActive = false;

  // Every cimrt_buffer allocated from this device, so cimrt_close can
  // orphan them instead of leaving a dangling back-pointer for the next
  // cimrt_write/cimrt_read/cimrt_copy to dereference. A raw non-owning set:
  // the buffers own their own lifetime via cimrt_free, this only needs to
  // find and null them out.
  std::unordered_set<cimrt_buffer *> liveBuffers;
};

struct cimrt_buffer {
  std::vector<uint8_t> data;
  cimrt_space space;
  // Back-pointer to the owning device, for cost accounting. Nulled by
  // cimrt_close when the device goes away first -- see the lifetime note on
  // cimrt_close in cimrt.h. Every write site that reads `dev` must treat
  // null as "orphaned", never assume it stays valid for the buffer's whole
  // lifetime.
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
  *out = nullptr;

  // A "-hw" suffix asks for real hardware, which is a different backend --
  // not this functional simulator. Reporting NO_DEVICE here is what makes
  // "no hardware attached" distinguishable from "simulator unavailable".
  const std::string name = target_name;
  if (name.size() >= 3 && name.compare(name.size() - 3, 3, "-hw") == 0)
    return cim::erbium::open(out);

  // A well-formed but adversarially large target spec (an absurd tile
  // count/geometry) could in principle exhaust memory during parsing or
  // construction; report it the same way an oversized cimrt_alloc is
  // reported rather than letting an allocator exception cross the
  // extern "C" boundary, which is undefined behaviour.
  cimrt_device *dev = nullptr;
  try {
    dev = new cimrt_device();
  } catch (const std::bad_alloc &) {
    return CIMRT_ERR_OOM;
  }
  dev->name = name;

  if (!parseTargetSpecFromFile(resolveTargetPath(name), dev->spec)) {
    delete dev;
    return CIMRT_ERR_NO_DEVICE;
  }

  // The target file may DESCRIBE a device this simulator cannot execute.
  // tiles.weight_dtype and tiles.activation_dtype are required fields that
  // nothing used to read -- their only consumers were two printfs in
  // cim-bench -- while cimrt_mvm below hardcodes the v0.1 contract (i8 x
  // i8 -> i32) in a comment and enforced it nowhere. So a vendor target
  // declaring `weight_dtype: i4` opened cleanly, compiled (the passes take
  // element types from the memrefs, never from the target), executed with
  // int8_t reinterpretation, and was charged full declared mvm cost:
  // plausible numbers for a different function than the file describes,
  // which is exactly what this project refuses everywhere else. It is also
  // the sharpest counterexample to "retargetable" available, since
  // precision is the axis a CIM vendor is most likely to differ on.
  //
  // Refused HERE rather than in the YAML parser on purpose. Parsing and
  // executability are different questions: `cim-bench dump-target` is an
  // inspection tool and must keep working on any well-formed file,
  // including one describing hardware this build cannot run -- being able
  // to read a vendor's spec before you can execute it is useful, not an
  // error. (Putting this in the parser also broke
  // test/python/test_yaml_differential.py for a real reason: schema.py
  // generates i4/u8/i16 as valid SPELLINGS, and that test compares the
  // reader against PyYAML on document syntax, not on semantic support.)
  // cimrt_open is where execution begins, so it is where "I cannot run
  // this" belongs.
  for (const auto &[field, value] :
       {std::pair<const char *, const std::string &>{"tiles.weight_dtype",
                                                     dev->spec.tiles.weightDtype},
        std::pair<const char *, const std::string &>{
            "tiles.activation_dtype", dev->spec.tiles.activationDtype}}) {
    if (value != "i8") {
      std::fprintf(stderr,
                   "cimrt_open: target '%s' declares %s: '%s', but this "
                   "functional simulator computes i8 x i8 -> i32 only "
                   "(cimrt_mvm) and would reinterpret any other width as "
                   "i8 rather than refuse it\n",
                   name.c_str(), field, value.c_str());
      delete dev;
      return CIMRT_ERR_NO_DEVICE;
    }
  }

  *out = dev;
  return CIMRT_OK;
}

void cimrt_close(cimrt_device *dev) {
  if (!dev)
    return;
  // Orphan every buffer still allocated from this device -- see the
  // lifetime note on cimrt_close in cimrt.h. `liveBuffers` is exactly the
  // set of buffers whose `dev` still points here, so this cannot miss one
  // or double-orphan one already freed (cimrt_free deregisters first).
  for (cimrt_buffer *buf : dev->liveBuffers)
    buf->dev = nullptr;
  delete dev;
}

cimrt_status cimrt_query(cimrt_device *dev, cimrt_device_info *out) {
  if (!dev || !out)
    return CIMRT_ERR_INVALID_ARG;
  std::memset(out, 0, sizeof(*out));
  std::strncpy(out->name, dev->name.c_str(), sizeof(out->name) - 1);
  out->num_tiles = dev->spec.tiles.count;
  out->tile_rows = dev->spec.tiles.rows;
  out->tile_cols = dev->spec.tiles.cols;
  out->persistent = dev->spec.tiles.persistent;
  out->partial_sum_in_place = dev->spec.capabilities.partialSumInPlace;
  out->max_in_place = dev->spec.capabilities.maxInPlace;
  return CIMRT_OK;
}

cimrt_status cimrt_alloc(cimrt_device *dev, size_t bytes, cimrt_space space,
                          cimrt_buffer **out) {
  if (!dev || !out)
    return CIMRT_ERR_INVALID_ARG;
  *out = nullptr;

  cimrt_buffer *buf = nullptr;
  try {
    buf = new cimrt_buffer();
    buf->data.resize(bytes, 0);
  } catch (const std::bad_alloc &) {
    delete buf;
    return CIMRT_ERR_OOM;
  } catch (const std::length_error &) {
    // vector::resize throws this for a size past its max_size(), which is
    // an unsatisfiable request rather than a transient allocator failure,
    // but it is the same answer from the caller's side: cannot have it.
    delete buf;
    return CIMRT_ERR_OOM;
  }
  buf->space = space;
  buf->dev = dev;
  dev->liveBuffers.insert(buf);
  *out = buf;
  return CIMRT_OK;
}

cimrt_status cimrt_copy(cimrt_buffer *dst, const cimrt_buffer *src) {
  if (!dst || !src)
    return CIMRT_ERR_INVALID_ARG;
  if (dst->data.size() != src->data.size())
    return CIMRT_ERR_SHAPE_MISMATCH;
  // An orphaned destination (its device already closed) cannot have this
  // transfer costed, and costing it against nothing would make the cost
  // model silently wrong rather than loudly unavailable -- see the
  // cimrt_close lifetime note in cimrt.h.
  if (!dst->dev)
    return CIMRT_ERR_INVALID_ARG;
  dst->data = src->data;
  dst->dev->cost.recordTransfer(dst->data.size());
  return CIMRT_OK;
}

cimrt_status cimrt_copy_range(cimrt_buffer *dst, size_t dst_offset,
                              const cimrt_buffer *src, size_t src_offset,
                              size_t bytes) {
  if (!dst || !src)
    return CIMRT_ERR_INVALID_ARG;
  if (dst == src)
    return CIMRT_ERR_INVALID_ARG; // cimrt.h: dst and src must differ.
  if (!dst->dev)
    return CIMRT_ERR_INVALID_ARG; // orphaned -- see cimrt_close's note
  // Checked against overflow rather than just `offset + bytes > size`,
  // which wraps for adversarial offsets -- same guard cimrt_write/
  // cimrt_read already use.
  if (dst_offset > dst->data.size() || bytes > dst->data.size() - dst_offset)
    return CIMRT_ERR_INVALID_ARG;
  if (src_offset > src->data.size() || bytes > src->data.size() - src_offset)
    return CIMRT_ERR_INVALID_ARG;
  std::memcpy(dst->data.data() + dst_offset, src->data.data() + src_offset,
              bytes);
  dst->dev->cost.recordTransfer(bytes);
  return CIMRT_OK;
}

cimrt_status cimrt_write(cimrt_buffer *dst, size_t dst_offset, const void *src,
                          size_t bytes) {
  if (!dst || !src)
    return CIMRT_ERR_INVALID_ARG;
  if (!dst->dev)
    return CIMRT_ERR_INVALID_ARG; // orphaned -- see cimrt_close's note
  // Checked against overflow rather than just `offset + bytes > size`,
  // which wraps for adversarial offsets.
  if (dst_offset > dst->data.size() || bytes > dst->data.size() - dst_offset)
    return CIMRT_ERR_INVALID_ARG;
  std::memcpy(dst->data.data() + dst_offset, src, bytes);
  dst->dev->cost.recordTransfer(bytes);
  return CIMRT_OK;
}

cimrt_status cimrt_read(const cimrt_buffer *src, size_t src_offset, void *dst,
                         size_t bytes) {
  if (!src || !dst)
    return CIMRT_ERR_INVALID_ARG;
  if (!src->dev)
    return CIMRT_ERR_INVALID_ARG; // orphaned -- see cimrt_close's note
  if (src_offset > src->data.size() || bytes > src->data.size() - src_offset)
    return CIMRT_ERR_INVALID_ARG;
  std::memcpy(dst, src->data.data() + src_offset, bytes);
  src->dev->cost.recordTransfer(bytes);
  return CIMRT_OK;
}

void cimrt_free(cimrt_buffer *buf) {
  if (!buf)
    return;
  // Deregister before deleting: a buffer freed while its device is still
  // open must not leave a dangling entry in dev->liveBuffers for the next
  // cimrt_close to walk and write through -- the mirror image of the
  // use-after-free this registry exists to prevent.
  if (buf->dev)
    buf->dev->liveBuffers.erase(buf);
  delete buf;
}

cimrt_status cimrt_program(cimrt_device *dev, cimrt_tile_id tile,
                            const cimrt_buffer *weights) {
  if (!dev || !weights)
    return CIMRT_ERR_INVALID_ARG;
  // A tile id the target does not have is a malformed call, not "busy":
  // TILE_BUSY says "this tile exists and is not ready", which sends a
  // caller looking for a scheduling problem that is not there.
  if (tile >= dev->spec.tiles.count)
    return CIMRT_ERR_INVALID_ARG;
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
  // Same distinction as cimrt_program: a tile id the target does not have
  // is malformed, not "busy". Must be checked before the tileWeights
  // lookup below, which cannot otherwise tell "out of range" apart from
  // "in range but never programmed".
  if (tile >= dev->spec.tiles.count)
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

/// Whole bytes for `bits`, or 0 if `bits` is not a positive multiple of 8 --
/// this ABI is exactly as byte-addressed as cimrt_alloc/write/read
/// (and Interpreter.cpp's elementByteWidth, which this mirrors).
static uint32_t bitsToBytes(uint32_t bits) {
  return (bits > 0 && bits % 8 == 0) ? bits / 8 : 0;
}

/// Sign-extend the `bytes`-byte little-endian integer starting at `p` to
/// int64_t. Mirrors Interpreter.cpp's runRequantize sign-extension exactly
/// (same reasoning: a bare cast would read the high bit as magnitude, not
/// sign, for anything narrower than int64_t).
static int64_t signExtend(const uint8_t *p, uint32_t bytes) {
  uint64_t raw = 0;
  std::memcpy(&raw, p, bytes);
  const uint32_t bits = bytes * 8;
  if (bits >= 64)
    return static_cast<int64_t>(raw);
  const uint64_t signBit = static_cast<uint64_t>(1) << (bits - 1);
  return static_cast<int64_t>((raw ^ signBit) - signBit);
}

cimrt_status cimrt_requantize(cimrt_device *dev, const cimrt_buffer *input,
                               cimrt_buffer *output, size_t count,
                               uint32_t in_bits, uint32_t out_bits,
                               float scale, int32_t zero_point,
                               uint32_t effective_bits) {
  if (!dev || !input || !output)
    return CIMRT_ERR_INVALID_ARG;
  const uint32_t inBytes = bitsToBytes(in_bits);
  const uint32_t outBytes = bitsToBytes(out_bits);
  if (inBytes == 0 || outBytes == 0)
    return CIMRT_ERR_INVALID_ARG;
  if (effective_bits == 0 || effective_bits > out_bits)
    return CIMRT_ERR_INVALID_ARG;
  if (!(scale > 0.0f))
    return CIMRT_ERR_INVALID_ARG;
  if (input->data.size() != count * inBytes ||
      output->data.size() != count * outBytes)
    return CIMRT_ERR_SHAPE_MISMATCH;

  // The SIGNED range effective_bits can hold, regardless of whether
  // out_bits is nominally wider -- the readout path this op stands for
  // (an analog ADC's real resolution, or a digital requantizer's narrower
  // accumulator) is what actually limits precision, not the container.
  const int64_t clampMin = -(static_cast<int64_t>(1) << (effective_bits - 1));
  const int64_t clampMax = (static_cast<int64_t>(1) << (effective_bits - 1)) - 1;

  for (size_t i = 0; i < count; ++i) {
    const int64_t value = signExtend(input->data.data() + i * inBytes, inBytes);
    // round-half-away-from-zero, the one mode that treats positive and
    // negative accumulator values symmetrically. This is the sole
    // implementation of cim.requantize's arithmetic: Interpreter.cpp's
    // runRequantize calls this function directly (mirroring how it already
    // delegates cim.program/cim.mvm to cimrt_program/cimrt_mvm) rather than
    // recomputing it host-side, so the cost accounting below and the
    // arithmetic can never drift apart the way two independent copies
    // could. test/mlir/legalize_precision_e2e_test.cpp's expectedRequantize
    // is the independent third implementation that checks this one.
    const double scaled = static_cast<double>(value) / static_cast<double>(scale);
    int64_t quantized = static_cast<int64_t>(zero_point) +
                        static_cast<int64_t>(std::llround(scaled));
    quantized = std::clamp(quantized, clampMin, clampMax);

    const uint64_t bits = static_cast<uint64_t>(quantized);
    std::memcpy(output->data.data() + i * outBytes, &bits, outBytes);
  }

  dev->cost.recordRequantize();
  return CIMRT_OK;
}

cimrt_status cimrt_reduce_add(cimrt_device *dev, cimrt_buffer *out,
                              const cimrt_buffer *a, const cimrt_buffer *b,
                              size_t count, uint32_t bits) {
  if (!dev || !out || !a || !b)
    return CIMRT_ERR_INVALID_ARG;
  if (out == a || out == b)
    return CIMRT_ERR_INVALID_ARG; // cimrt.h: out must not alias a or b.
  const uint32_t bytes = bitsToBytes(bits);
  if (bytes == 0)
    return CIMRT_ERR_INVALID_ARG;
  if (a->data.size() != count * bytes || b->data.size() != count * bytes ||
      out->data.size() != count * bytes)
    return CIMRT_ERR_SHAPE_MISMATCH;

  for (size_t i = 0; i < count; ++i) {
    uint64_t lhs = 0, rhs = 0;
    std::memcpy(&lhs, a->data.data() + i * bytes, bytes);
    std::memcpy(&rhs, b->data.data() + i * bytes, bytes);
    // Wrapping addition, matching Interpreter.cpp's runReducePartial: a
    // fixed-width accumulator wraps, it does not saturate. Widened to
    // uint64_t so the add itself never overflows a C++ type before the
    // result is truncated back to `bytes` -- correct for bytes up to 8;
    // this ABI's own byte-addressed contract (bitsToBytes) never produces
    // more than that from a 32-bit `bits` parameter in practice, but the
    // truncating memcpy below is what actually enforces the wrap either
    // way, not this widening.
    const uint64_t sum = lhs + rhs;
    std::memcpy(out->data.data() + i * bytes, &sum, bytes);
  }

  dev->cost.recordReduceAdd();
  return CIMRT_OK;
}

cimrt_status cimrt_reduce_add_inplace(cimrt_device *dev, cimrt_buffer *acc,
                                      const cimrt_buffer *rhs, size_t count,
                                      uint32_t bits) {
  if (!dev || !acc || !rhs)
    return CIMRT_ERR_INVALID_ARG;
  if (acc == rhs)
    return CIMRT_ERR_INVALID_ARG; // cimrt.h: not a shape cim.reduce_partial
                                  // ever produces.
  const uint32_t bytes = bitsToBytes(bits);
  if (bytes == 0)
    return CIMRT_ERR_INVALID_ARG;
  if (acc->data.size() != count * bytes || rhs->data.size() != count * bytes)
    return CIMRT_ERR_SHAPE_MISMATCH;

  for (size_t i = 0; i < count; ++i) {
    uint64_t lhs = 0, r = 0;
    std::memcpy(&lhs, acc->data.data() + i * bytes, bytes);
    std::memcpy(&r, rhs->data.data() + i * bytes, bytes);
    // Same wrapping-add contract as cimrt_reduce_add above, just folded
    // into the first operand's own storage instead of a third buffer.
    const uint64_t sum = lhs + r;
    std::memcpy(acc->data.data() + i * bytes, &sum, bytes);
  }

  dev->cost.recordReduceAdd();
  return CIMRT_OK;
}

cimrt_status cimrt_reduce_max(cimrt_device *dev, cimrt_buffer *out,
                              const cimrt_buffer *a, const cimrt_buffer *b,
                              size_t count, uint32_t bits) {
  if (!dev || !out || !a || !b)
    return CIMRT_ERR_INVALID_ARG;
  if (out == a || out == b)
    return CIMRT_ERR_INVALID_ARG; // cimrt.h: out must not alias a or b.
  const uint32_t bytes = bitsToBytes(bits);
  if (bytes == 0)
    return CIMRT_ERR_INVALID_ARG;
  if (a->data.size() != count * bytes || b->data.size() != count * bytes ||
      out->data.size() != count * bytes)
    return CIMRT_ERR_SHAPE_MISMATCH;

  for (size_t i = 0; i < count; ++i) {
    // SIGN-EXTEND, then compare as int64_t. This is the one line where this
    // function must NOT follow cimrt_reduce_add's lead: that function reads
    // both operands as raw uint64_t because a wrapping add is bit-identical
    // either way, but a maximum is not. As unsigned bytes an int8 -1 is
    // 0xFF and would beat 5; ONNX MaxPool on int8 compares signed and
    // returns 5. Same helper cimrt_requantize uses, for the same reason.
    const int64_t lhs = signExtend(a->data.data() + i * bytes, bytes);
    const int64_t rhs = signExtend(b->data.data() + i * bytes, bytes);
    const int64_t winner = lhs > rhs ? lhs : rhs;

    // Truncating store, matching cimrt_reduce_add's own: `winner` is one of
    // the two inputs unchanged, so it always fits `bytes` and this narrows
    // nothing -- unlike an add, a max cannot leave its operands' range.
    const uint64_t raw = static_cast<uint64_t>(winner);
    std::memcpy(out->data.data() + i * bytes, &raw, bytes);
  }

  dev->cost.recordReduceMax();
  return CIMRT_OK;
}

cimrt_status cimrt_reduce_max_inplace(cimrt_device *dev, cimrt_buffer *acc,
                                      const cimrt_buffer *rhs, size_t count,
                                      uint32_t bits) {
  if (!dev || !acc || !rhs)
    return CIMRT_ERR_INVALID_ARG;
  if (acc == rhs)
    return CIMRT_ERR_INVALID_ARG; // cimrt.h: not a shape cim.reduce_max
                                  // ever produces.
  const uint32_t bytes = bitsToBytes(bits);
  if (bytes == 0)
    return CIMRT_ERR_INVALID_ARG;
  if (acc->data.size() != count * bytes || rhs->data.size() != count * bytes)
    return CIMRT_ERR_SHAPE_MISMATCH;

  for (size_t i = 0; i < count; ++i) {
    // Same sign-extending compare as cimrt_reduce_max above, for the same
    // reason: a raw-byte compare would read an int8 -1 as 0xFF and beat 5.
    const int64_t lhs = signExtend(acc->data.data() + i * bytes, bytes);
    const int64_t r = signExtend(rhs->data.data() + i * bytes, bytes);
    const int64_t winner = lhs > r ? lhs : r;

    // Truncating store into the first operand's own storage instead of a
    // third buffer -- the same fold cimrt_reduce_add_inplace does, and
    // `winner` always fits `bytes` for the same reason cimrt_reduce_max's
    // own store does: it is one of the two inputs unchanged.
    const uint64_t raw = static_cast<uint64_t>(winner);
    std::memcpy(acc->data.data() + i * bytes, &raw, bytes);
  }

  dev->cost.recordReduceMax();
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
  dev->profileBaseline = dev->cost.snapshot();
  dev->profilingActive = true;
  return CIMRT_OK;
}

cimrt_status cimrt_profile_stop(cimrt_device *dev, cimrt_profile *out) {
  if (!dev || !out)
    return CIMRT_ERR_INVALID_ARG;
  // A stop with no matching start reports zero, not the cumulative total
  // since cimrt_open (see the contract comment on cimrt_profile_start/stop
  // in cimrt.h): baseline == current counters gives a zero delta below,
  // which is exactly that "no window was opened" answer.
  const CostAccumulator::Snapshot baseline =
      dev->profilingActive ? dev->profileBaseline : dev->cost.snapshot();
  const CostAccumulator::Snapshot now = dev->cost.snapshot();
  out->programs_issued = now.programsIssued - baseline.programsIssued;
  out->mvms_issued = now.mvmsIssued - baseline.mvmsIssued;
  out->requantizes_issued = now.requantizesIssued - baseline.requantizesIssued;
  out->reduce_adds_issued = now.reduceAddsIssued - baseline.reduceAddsIssued;
  out->reduce_maxes_issued =
      now.reduceMaxesIssued - baseline.reduceMaxesIssued;
  out->bytes_transferred = now.bytesTransferred - baseline.bytesTransferred;
  out->estimated_energy_pj = now.energyPj - baseline.energyPj;
  out->estimated_latency_ns = now.latencyNs - baseline.latencyNs;
  dev->profilingActive = false;
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
