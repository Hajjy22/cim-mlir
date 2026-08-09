//===- cimrt_test.cpp - Tests for the cimrt C API --------------*- C++ -*-===//
//
// Before this file existed, cimrt was linked into no test binary at all and
// the INT8 matrix-vector multiply in runtime/src/simulator/simulator.cpp had
// never executed. That was not an oversight in the test suite: the API had
// no host->buffer write path, so no caller could put a nonzero byte into a
// buffer and no caller could ever observe a nonzero result. cimrt_write and
// cimrt_read exist to close that hole, and these tests are what prove the
// arithmetic underneath was ever right.
//
//===----------------------------------------------------------------------===//

#include "test_harness.h"

#include "cimrt.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

/// The tiny 2-tile 4x4 target. Located relative to the source tree so the
/// binary can be run from the build directory or the repo root.
const char *tinyTargetPath() {
  static std::string path = [] {
    for (const char *candidate :
         {"test/targets/tiny-4x4.yaml", "../test/targets/tiny-4x4.yaml",
          "../../test/targets/tiny-4x4.yaml"}) {
      cimrt_device *probe = nullptr;
      if (cimrt_open(candidate, &probe) == CIMRT_OK) {
        cimrt_close(probe);
        return std::string(candidate);
      }
    }
    return std::string("test/targets/tiny-4x4.yaml");
  }();
  return path.c_str();
}

/// RAII device so a failed assertion cannot leak under ASan/valgrind.
struct Device {
  cimrt_device *dev = nullptr;
  explicit Device(const char *target = nullptr) {
    status = cimrt_open(target ? target : tinyTargetPath(), &dev);
  }
  ~Device() { cimrt_close(dev); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  cimrt_status status = CIMRT_ERR_NO_DEVICE;
};

struct Buffer {
  cimrt_buffer *buf = nullptr;
  ~Buffer() { cimrt_free(buf); }
  Buffer() = default;
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
};

/// Reference matrix-vector multiply: out[r] = sum_c W[r][c] * act[c].
std::vector<int32_t> referenceMvm(const std::vector<int8_t> &w,
                                   const std::vector<int8_t> &act, int rows,
                                   int cols) {
  std::vector<int32_t> out(rows, 0);
  for (int r = 0; r < rows; ++r) {
    int64_t acc = 0;
    for (int c = 0; c < cols; ++c)
      acc += static_cast<int64_t>(w[r * cols + c]) * act[c];
    out[r] = static_cast<int32_t>(static_cast<uint32_t>(acc));
  }
  return out;
}

/// Independent reference for cimrt_requantize's arithmetic -- the same
/// formula documented in cimrt.h and implemented in simulator.cpp,
/// computed here a second, independent way (not by calling into the
/// implementation under test) so a bug shared between the two would not be
/// invisible to this test: round-half-away-from-zero, then clamp to the
/// signed range effective_bits can hold.
int64_t expectedRequantize(int64_t value, float scale, int32_t zeroPoint,
                            uint32_t effectiveBits) {
  const double scaled =
      static_cast<double>(value) / static_cast<double>(scale);
  int64_t quantized =
      static_cast<int64_t>(zeroPoint) + static_cast<int64_t>(std::llround(scaled));
  const int64_t clampMin = -(static_cast<int64_t>(1) << (effectiveBits - 1));
  const int64_t clampMax = (static_cast<int64_t>(1) << (effectiveBits - 1)) - 1;
  if (quantized < clampMin)
    quantized = clampMin;
  if (quantized > clampMax)
    quantized = clampMax;
  return quantized;
}

/// Packs `values` as `bytes`-wide little-endian signed integers (matching
/// this platform's native layout, the same assumption every raw-byte path
/// in this project already makes) for cimrt_write.
std::vector<uint8_t> packSigned(const std::vector<int64_t> &values,
                                 uint32_t bytes) {
  std::vector<uint8_t> out(values.size() * bytes);
  for (size_t i = 0; i < values.size(); ++i) {
    uint64_t bits = static_cast<uint64_t>(values[i]);
    std::memcpy(out.data() + i * bytes, &bits, bytes);
  }
  return out;
}

/// Inverse of packSigned, sign-extending each `bytes`-wide element back to
/// int64_t -- for reading a cimrt_requantize result back out regardless of
/// its container width.
std::vector<int64_t> unpackSigned(const std::vector<uint8_t> &raw,
                                   uint32_t bytes) {
  std::vector<int64_t> out(raw.size() / bytes);
  for (size_t i = 0; i < out.size(); ++i) {
    uint64_t bits = 0;
    std::memcpy(&bits, raw.data() + i * bytes, bytes);
    const uint32_t width = bytes * 8;
    if (width < 64) {
      const uint64_t signBit = static_cast<uint64_t>(1) << (width - 1);
      bits = (bits ^ signBit) - signBit;
    }
    out[i] = static_cast<int64_t>(bits);
  }
  return out;
}

} // namespace

//===----------------------------------------------------------------------===//
// Device lifecycle
//===----------------------------------------------------------------------===//

CIM_TEST(cimrt_opens_a_target_by_path) {
  Device dev;
  CIM_EXPECT_EQ(dev.status, CIMRT_OK);
  CIM_EXPECT(dev.dev != nullptr);

  cimrt_device_info info{};
  CIM_EXPECT_EQ(cimrt_query(dev.dev, &info), CIMRT_OK);
  CIM_EXPECT_EQ(info.num_tiles, 2u);
  CIM_EXPECT_EQ(info.tile_rows, 4u);
  CIM_EXPECT_EQ(info.tile_cols, 4u);
  CIM_EXPECT(info.persistent);
  CIM_EXPECT_CONTAINS(std::string(info.name), "tiny-4x4");
}

CIM_TEST(cimrt_rejects_a_missing_target) {
  cimrt_device *dev = nullptr;
  CIM_EXPECT_EQ(cimrt_open("test/targets/no-such-target.yaml", &dev),
                CIMRT_ERR_NO_DEVICE);
  CIM_EXPECT(dev == nullptr);
}

CIM_TEST(cimrt_hardware_target_reports_no_device_not_a_simulator) {
  // The "-hw" suffix routes to the real hardware backend. Answering
  // NO_DEVICE is the truthful answer -- and, importantly, it must NOT
  // silently fall back to the functional simulator, which would make a
  // test think it had exercised hardware.
  cimrt_device *dev = nullptr;
  CIM_EXPECT_EQ(cimrt_open("erbium-8t-hw", &dev), CIMRT_ERR_NO_DEVICE);
  CIM_EXPECT(dev == nullptr);
}

CIM_TEST(cimrt_close_and_free_tolerate_null) {
  // Called on every error path in the RAII wrappers above.
  cimrt_close(nullptr);
  cimrt_free(nullptr);
}

//===----------------------------------------------------------------------===//
// The host <-> buffer data path
//===----------------------------------------------------------------------===//

CIM_TEST(cimrt_write_then_read_round_trips) {
  Device dev;
  CIM_EXPECT_EQ(dev.status, CIMRT_OK);

  Buffer b;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &b.buf), CIMRT_OK);

  const std::vector<uint8_t> in = {1, 2, 3, 4, 5, 6, 7, 8};
  CIM_EXPECT_EQ(cimrt_write(b.buf, 0, in.data(), in.size()), CIMRT_OK);

  std::vector<uint8_t> out(8, 0);
  CIM_EXPECT_EQ(cimrt_read(b.buf, 0, out.data(), out.size()), CIMRT_OK);
  CIM_EXPECT(in == out);
}

CIM_TEST(cimrt_alloc_zero_fills) {
  Device dev;
  Buffer b;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &b.buf), CIMRT_OK);
  std::vector<uint8_t> out(4, 0xFF);
  CIM_EXPECT_EQ(cimrt_read(b.buf, 0, out.data(), out.size()), CIMRT_OK);
  for (uint8_t byte : out)
    CIM_EXPECT_EQ(static_cast<int>(byte), 0);
}

CIM_TEST(cimrt_write_and_read_honour_offsets) {
  Device dev;
  Buffer b;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &b.buf), CIMRT_OK);

  const std::vector<uint8_t> mid = {0xAA, 0xBB};
  CIM_EXPECT_EQ(cimrt_write(b.buf, 3, mid.data(), mid.size()), CIMRT_OK);

  std::vector<uint8_t> all(8, 0);
  CIM_EXPECT_EQ(cimrt_read(b.buf, 0, all.data(), all.size()), CIMRT_OK);
  CIM_EXPECT_EQ(static_cast<int>(all[2]), 0);
  CIM_EXPECT_EQ(static_cast<int>(all[3]), 0xAA);
  CIM_EXPECT_EQ(static_cast<int>(all[4]), 0xBB);
  CIM_EXPECT_EQ(static_cast<int>(all[5]), 0);
}

CIM_TEST(cimrt_write_and_read_reject_out_of_range) {
  Device dev;
  Buffer b;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &b.buf), CIMRT_OK);

  const std::vector<uint8_t> src = {1, 2, 3, 4};
  std::vector<uint8_t> dst(4, 0);

  CIM_EXPECT_EQ(cimrt_write(b.buf, 1, src.data(), 4), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_write(b.buf, 5, src.data(), 1), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_read(b.buf, 1, dst.data(), 4), CIMRT_ERR_INVALID_ARG);

  // An offset near SIZE_MAX must be rejected by the range check rather than
  // wrapping into a valid-looking one.
  const size_t huge = static_cast<size_t>(-1);
  CIM_EXPECT_EQ(cimrt_write(b.buf, huge, src.data(), 4), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_read(b.buf, huge, dst.data(), 4), CIMRT_ERR_INVALID_ARG);
}

CIM_TEST(cimrt_copy_requires_matching_sizes) {
  Device dev;
  Buffer a, b, small;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &a.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &b.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 2, CIMRT_SPACE_NEAR, &small.buf), CIMRT_OK);

  const std::vector<uint8_t> in = {9, 8, 7, 6};
  CIM_EXPECT_EQ(cimrt_write(a.buf, 0, in.data(), in.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_copy(b.buf, a.buf), CIMRT_OK);

  std::vector<uint8_t> out(4, 0);
  CIM_EXPECT_EQ(cimrt_read(b.buf, 0, out.data(), out.size()), CIMRT_OK);
  CIM_EXPECT(in == out);

  CIM_EXPECT_EQ(cimrt_copy(small.buf, a.buf), CIMRT_ERR_SHAPE_MISMATCH);
}

//===----------------------------------------------------------------------===//
// requantize: cim-lower-to-target's real device-side readout/clamp
//===----------------------------------------------------------------------===//

CIM_TEST(cimrt_requantize_matches_hand_computed_values) {
  Device dev;
  CIM_EXPECT_EQ(dev.status, CIMRT_OK);

  // Same width in and out (i32 -> i32, cim-legalize-precision's
  // width-preserving fallback shape): values chosen to exercise the clamp
  // in both directions and to leave one value untouched.
  const std::vector<int64_t> values = {1000, -1000, 0, 50};
  Buffer in, out;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, values.size() * 4, CIMRT_SPACE_NEAR,
                            &in.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, values.size() * 4, CIMRT_SPACE_NEAR,
                            &out.buf),
                CIMRT_OK);
  const std::vector<uint8_t> packed = packSigned(values, 4);
  CIM_EXPECT_EQ(cimrt_write(in.buf, 0, packed.data(), packed.size()), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, values.size(),
                                 /*in_bits=*/32, /*out_bits=*/32,
                                 /*scale=*/1.0f, /*zero_point=*/0,
                                 /*effective_bits=*/8),
                CIMRT_OK);

  std::vector<uint8_t> raw(values.size() * 4, 0);
  CIM_EXPECT_EQ(cimrt_read(out.buf, 0, raw.data(), raw.size()), CIMRT_OK);
  const std::vector<int64_t> got = unpackSigned(raw, 4);
  for (size_t i = 0; i < values.size(); ++i)
    CIM_EXPECT_EQ(got[i], expectedRequantize(values[i], 1.0f, 0, 8));
}

CIM_TEST(cimrt_requantize_narrows_the_output_container) {
  // i32 in, i8 out -- the shape cim-legalize-precision's normal
  // tile-native-precision narrowing produces.
  Device dev;
  Buffer in, out;
  const std::vector<int64_t> values = {127, -128, 5, -5};
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, values.size() * 4, CIMRT_SPACE_NEAR,
                            &in.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, values.size(), CIMRT_SPACE_NEAR, &out.buf),
                CIMRT_OK);
  const std::vector<uint8_t> packed = packSigned(values, 4);
  CIM_EXPECT_EQ(cimrt_write(in.buf, 0, packed.data(), packed.size()), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, values.size(),
                                 /*in_bits=*/32, /*out_bits=*/8,
                                 /*scale=*/1.0f, /*zero_point=*/0,
                                 /*effective_bits=*/8),
                CIMRT_OK);

  std::vector<uint8_t> raw(values.size(), 0);
  CIM_EXPECT_EQ(cimrt_read(out.buf, 0, raw.data(), raw.size()), CIMRT_OK);
  const std::vector<int64_t> got = unpackSigned(raw, 1);
  for (size_t i = 0; i < values.size(); ++i)
    CIM_EXPECT_EQ(got[i], expectedRequantize(values[i], 1.0f, 0, 8));
}

CIM_TEST(cimrt_requantize_widens_the_output_container) {
  // i8 in, i32 out, effective_bits narrower than the container -- the
  // clamp must key on effective_bits, not on out_bits.
  Device dev;
  Buffer in, out;
  const std::vector<int64_t> values = {40, -40, 10, -10};
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, values.size(), CIMRT_SPACE_NEAR, &in.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, values.size() * 4, CIMRT_SPACE_NEAR,
                            &out.buf),
                CIMRT_OK);
  const std::vector<uint8_t> packed = packSigned(values, 1);
  CIM_EXPECT_EQ(cimrt_write(in.buf, 0, packed.data(), packed.size()), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, values.size(),
                                 /*in_bits=*/8, /*out_bits=*/32,
                                 /*scale=*/1.0f, /*zero_point=*/0,
                                 /*effective_bits=*/6),
                CIMRT_OK);

  std::vector<uint8_t> raw(values.size() * 4, 0);
  CIM_EXPECT_EQ(cimrt_read(out.buf, 0, raw.data(), raw.size()), CIMRT_OK);
  const std::vector<int64_t> got = unpackSigned(raw, 4);
  for (size_t i = 0; i < values.size(); ++i)
    // effective_bits=6 clamps to [-32, 31]: 40->31, -40->-32, 10 and -10
    // are untouched.
    CIM_EXPECT_EQ(got[i], expectedRequantize(values[i], 1.0f, 0, 6));
}

CIM_TEST(cimrt_requantize_rounds_half_away_from_zero_with_scale_and_zero_point) {
  Device dev;
  Buffer in, out;
  const std::vector<int64_t> values = {5, -5, 7, -7}; // /2.0 -> .5 ties
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, values.size() * 4, CIMRT_SPACE_NEAR,
                            &in.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, values.size() * 4, CIMRT_SPACE_NEAR,
                            &out.buf),
                CIMRT_OK);
  const std::vector<uint8_t> packed = packSigned(values, 4);
  CIM_EXPECT_EQ(cimrt_write(in.buf, 0, packed.data(), packed.size()), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, values.size(),
                                 /*in_bits=*/32, /*out_bits=*/32,
                                 /*scale=*/2.0f, /*zero_point=*/3,
                                 /*effective_bits=*/8),
                CIMRT_OK);

  std::vector<uint8_t> raw(values.size() * 4, 0);
  CIM_EXPECT_EQ(cimrt_read(out.buf, 0, raw.data(), raw.size()), CIMRT_OK);
  const std::vector<int64_t> got = unpackSigned(raw, 4);
  for (size_t i = 0; i < values.size(); ++i)
    CIM_EXPECT_EQ(got[i], expectedRequantize(values[i], 2.0f, 3, 8));
  // Pin the actual numbers too, not just agreement with the reference --
  // 5/2=2.5 rounds away from zero to 3, -5/2=-2.5 rounds to -3, both then
  // offset by zero_point=3.
  CIM_EXPECT_EQ(got[0], 6);  // round(2.5)=3, +3
  CIM_EXPECT_EQ(got[1], 0);  // round(-2.5)=-3, +3
}

CIM_TEST(cimrt_requantize_rejects_invalid_arguments) {
  Device dev;
  Buffer in, out;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &in.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &out.buf), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_requantize(nullptr, in.buf, out.buf, 2, 32, 32, 1.0f, 0, 8),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, nullptr, out.buf, 2, 32, 32, 1.0f, 0, 8),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, nullptr, 2, 32, 32, 1.0f, 0, 8),
                CIMRT_ERR_INVALID_ARG);
  // Not a whole-byte width.
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, 2, 5, 32, 1.0f, 0, 8),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, 2, 32, 0, 1.0f, 0, 8),
                CIMRT_ERR_INVALID_ARG);
  // effective_bits must be positive and not exceed out_bits.
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, 2, 32, 32, 1.0f, 0, 0),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, 2, 32, 8, 1.0f, 0, 16),
                CIMRT_ERR_INVALID_ARG);
  // scale must be positive.
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, 2, 32, 32, 0.0f, 0, 8),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, 2, 32, 32, -1.0f, 0, 8),
                CIMRT_ERR_INVALID_ARG);
  // Buffer sizes must match count * bits/8 on each side: `in`/`out` are
  // both 8 bytes, so count=2 at 32 bits (8 bytes) fits but count=4 does not.
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, in.buf, out.buf, 4, 32, 32, 1.0f, 0, 8),
                CIMRT_ERR_SHAPE_MISMATCH);
}

//===----------------------------------------------------------------------===//
// reduce_add: cim-lower-to-target's device-side cim.reduce_partial lowering
//===----------------------------------------------------------------------===//

CIM_TEST(cimrt_reduce_add_matches_hand_computed_values) {
  Device dev;
  CIM_EXPECT_EQ(dev.status, CIMRT_OK);

  const std::vector<int64_t> lhs = {1, -1, 100, 0};
  const std::vector<int64_t> rhs = {2, -2, -50, 7};
  Buffer a, b, out;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, lhs.size() * 4, CIMRT_SPACE_NEAR, &a.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, rhs.size() * 4, CIMRT_SPACE_NEAR, &b.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, lhs.size() * 4, CIMRT_SPACE_NEAR, &out.buf),
                CIMRT_OK);
  const std::vector<uint8_t> packedA = packSigned(lhs, 4);
  const std::vector<uint8_t> packedB = packSigned(rhs, 4);
  CIM_EXPECT_EQ(cimrt_write(a.buf, 0, packedA.data(), packedA.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(b.buf, 0, packedB.data(), packedB.size()), CIMRT_OK);

  CIM_EXPECT_EQ(
      cimrt_reduce_add(dev.dev, out.buf, a.buf, b.buf, lhs.size(), /*bits=*/32),
      CIMRT_OK);

  std::vector<uint8_t> raw(lhs.size() * 4, 0);
  CIM_EXPECT_EQ(cimrt_read(out.buf, 0, raw.data(), raw.size()), CIMRT_OK);
  const std::vector<int64_t> got = unpackSigned(raw, 4);
  for (size_t i = 0; i < lhs.size(); ++i)
    CIM_EXPECT_EQ(got[i], lhs[i] + rhs[i]);
}

CIM_TEST(cimrt_reduce_add_wraps_on_overflow_rather_than_saturating) {
  // cim.reduce_partial sums a real hardware accumulator's partial results,
  // and a fixed-width accumulator wraps -- matching Interpreter.cpp's
  // runReducePartial exactly (same reasoning as cimrt_requantize's rounding
  // mode: two independent implementations of the same contract must agree
  // bit for bit, not just approximately).
  Device dev;
  const std::vector<int64_t> lhs = {2147483647, -2147483648, -1};
  const std::vector<int64_t> rhs = {1, -1, -2147483648};
  Buffer a, b, out;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, lhs.size() * 4, CIMRT_SPACE_NEAR, &a.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, rhs.size() * 4, CIMRT_SPACE_NEAR, &b.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, lhs.size() * 4, CIMRT_SPACE_NEAR, &out.buf),
                CIMRT_OK);
  const std::vector<uint8_t> packedA = packSigned(lhs, 4);
  const std::vector<uint8_t> packedB = packSigned(rhs, 4);
  CIM_EXPECT_EQ(cimrt_write(a.buf, 0, packedA.data(), packedA.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(b.buf, 0, packedB.data(), packedB.size()), CIMRT_OK);

  CIM_EXPECT_EQ(
      cimrt_reduce_add(dev.dev, out.buf, a.buf, b.buf, lhs.size(), /*bits=*/32),
      CIMRT_OK);

  std::vector<uint8_t> raw(lhs.size() * 4, 0);
  CIM_EXPECT_EQ(cimrt_read(out.buf, 0, raw.data(), raw.size()), CIMRT_OK);
  const std::vector<int64_t> got = unpackSigned(raw, 4);
  CIM_EXPECT_EQ(got[0], -2147483648LL); // INT32_MAX + 1 wraps to INT32_MIN
  CIM_EXPECT_EQ(got[1], 2147483647LL);  // INT32_MIN + (-1) wraps to INT32_MAX
  // -1 (0xFFFFFFFF) + INT32_MIN (0x80000000) = 0x17FFFFFFF, truncated to 32
  // bits is 0x7FFFFFFF = INT32_MAX.
  CIM_EXPECT_EQ(got[2], 2147483647LL);
}

CIM_TEST(cimrt_reduce_add_supports_narrower_element_widths) {
  // Not hardcoded to i32 -- an explicit `bits` parameter, same reasoning as
  // cimrt_requantize's in_bits/out_bits, even though today's real pipeline
  // only ever reduces i32 accumulators.
  Device dev;
  const std::vector<int64_t> lhs = {100, -128, 0};
  const std::vector<int64_t> rhs = {27, 0, -1};
  Buffer a, b, out;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, lhs.size(), CIMRT_SPACE_NEAR, &a.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, rhs.size(), CIMRT_SPACE_NEAR, &b.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, lhs.size(), CIMRT_SPACE_NEAR, &out.buf),
                CIMRT_OK);
  const std::vector<uint8_t> packedA = packSigned(lhs, 1);
  const std::vector<uint8_t> packedB = packSigned(rhs, 1);
  CIM_EXPECT_EQ(cimrt_write(a.buf, 0, packedA.data(), packedA.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(b.buf, 0, packedB.data(), packedB.size()), CIMRT_OK);

  CIM_EXPECT_EQ(
      cimrt_reduce_add(dev.dev, out.buf, a.buf, b.buf, lhs.size(), /*bits=*/8),
      CIMRT_OK);

  std::vector<uint8_t> raw(lhs.size(), 0);
  CIM_EXPECT_EQ(cimrt_read(out.buf, 0, raw.data(), raw.size()), CIMRT_OK);
  const std::vector<int64_t> got = unpackSigned(raw, 1);
  CIM_EXPECT_EQ(got[0], 127);  // 100 + 27, fits in i8
  CIM_EXPECT_EQ(got[1], -128); // -128 + 0
  CIM_EXPECT_EQ(got[2], -1);   // 0 + (-1)
}

CIM_TEST(cimrt_reduce_add_rejects_invalid_arguments) {
  Device dev;
  Buffer a, b, out;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &a.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &b.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &out.buf), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_reduce_add(nullptr, out.buf, a.buf, b.buf, 2, 32),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_reduce_add(dev.dev, nullptr, a.buf, b.buf, 2, 32),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_reduce_add(dev.dev, out.buf, nullptr, b.buf, 2, 32),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_reduce_add(dev.dev, out.buf, a.buf, nullptr, 2, 32),
                CIMRT_ERR_INVALID_ARG);
  // out must not alias either input.
  CIM_EXPECT_EQ(cimrt_reduce_add(dev.dev, a.buf, a.buf, b.buf, 2, 32),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_reduce_add(dev.dev, b.buf, a.buf, b.buf, 2, 32),
                CIMRT_ERR_INVALID_ARG);
  // Not a whole-byte width.
  CIM_EXPECT_EQ(cimrt_reduce_add(dev.dev, out.buf, a.buf, b.buf, 2, 5),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_reduce_add(dev.dev, out.buf, a.buf, b.buf, 2, 0),
                CIMRT_ERR_INVALID_ARG);
  // Buffer sizes must match count * bits/8: all three buffers are 8 bytes,
  // so count=2 at 32 bits fits but count=4 does not.
  CIM_EXPECT_EQ(cimrt_reduce_add(dev.dev, out.buf, a.buf, b.buf, 4, 32),
                CIMRT_ERR_SHAPE_MISMATCH);
}

//===----------------------------------------------------------------------===//
// copy_range: cim-lower-to-target's device-side non-identity subview
// materialization (a genuine slice of a device-space buffer into a fresh
// one of the slice's own size)
//===----------------------------------------------------------------------===//

CIM_TEST(cimrt_copy_range_matches_hand_computed_values) {
  Device dev;
  CIM_EXPECT_EQ(dev.status, CIMRT_OK);

  const std::vector<int64_t> values = {10, 20, 30, 40, 50, 60};
  Buffer src, dst;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, values.size(), CIMRT_SPACE_NEAR, &src.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 3, CIMRT_SPACE_NEAR, &dst.buf), CIMRT_OK);
  const std::vector<uint8_t> packed = packSigned(values, 1);
  CIM_EXPECT_EQ(cimrt_write(src.buf, 0, packed.data(), packed.size()), CIMRT_OK);

  // The middle 3 elements: [20, 30, 40] at byte offset 1.
  CIM_EXPECT_EQ(cimrt_copy_range(dst.buf, 0, src.buf, 1, 3), CIMRT_OK);

  std::vector<uint8_t> raw(3, 0);
  CIM_EXPECT_EQ(cimrt_read(dst.buf, 0, raw.data(), raw.size()), CIMRT_OK);
  const std::vector<int64_t> got = unpackSigned(raw, 1);
  CIM_EXPECT_EQ(got[0], 20);
  CIM_EXPECT_EQ(got[1], 30);
  CIM_EXPECT_EQ(got[2], 40);
}

CIM_TEST(cimrt_copy_range_writes_at_a_nonzero_destination_offset) {
  Device dev;
  Buffer src, dst;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &src.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &dst.buf), CIMRT_OK);
  const std::vector<int64_t> values = {1, 2, 3, 4};
  const std::vector<uint8_t> packed = packSigned(values, 1);
  CIM_EXPECT_EQ(cimrt_write(src.buf, 0, packed.data(), packed.size()), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_copy_range(dst.buf, 4, src.buf, 0, 4), CIMRT_OK);

  std::vector<uint8_t> raw(8, 0);
  CIM_EXPECT_EQ(cimrt_read(dst.buf, 0, raw.data(), raw.size()), CIMRT_OK);
  const std::vector<int64_t> got = unpackSigned(raw, 1);
  for (int i = 0; i < 4; ++i)
    CIM_EXPECT_EQ(got[i], 0); // untouched leading region
  for (int i = 0; i < 4; ++i)
    CIM_EXPECT_EQ(got[4 + i], values[i]);
}

CIM_TEST(cimrt_copy_range_rejects_invalid_arguments) {
  Device dev;
  Buffer src, dst;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &src.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 8, CIMRT_SPACE_NEAR, &dst.buf), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_copy_range(nullptr, 0, src.buf, 0, 4),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_copy_range(dst.buf, 0, nullptr, 0, 4),
                CIMRT_ERR_INVALID_ARG);
  // dst and src must differ.
  CIM_EXPECT_EQ(cimrt_copy_range(src.buf, 0, src.buf, 0, 4),
                CIMRT_ERR_INVALID_ARG);
  // Out-of-range destination and source windows, including the
  // overflow-safe form (offset within bounds, but offset + bytes is not).
  CIM_EXPECT_EQ(cimrt_copy_range(dst.buf, 9, src.buf, 0, 1),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_copy_range(dst.buf, 6, src.buf, 0, 4),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_copy_range(dst.buf, 0, src.buf, 9, 1),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_copy_range(dst.buf, 0, src.buf, 6, 4),
                CIMRT_ERR_INVALID_ARG);
}

CIM_TEST(cimrt_copy_range_reports_orphaned_destination) {
  // Matching cimrt_copy/cimrt_write's own orphan handling -- see
  // cimrt_close's lifetime note in cimrt.h.
  Buffer dst, src;
  {
    Device dev;
    CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &dst.buf), CIMRT_OK);
    CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &src.buf), CIMRT_OK);
  } // dev closes here, orphaning both buffers.
  CIM_EXPECT_EQ(cimrt_copy_range(dst.buf, 0, src.buf, 0, 4),
                CIMRT_ERR_INVALID_ARG);
}

//===----------------------------------------------------------------------===//
// program + mvm: the arithmetic that had never run
//===----------------------------------------------------------------------===//

CIM_TEST(cimrt_mvm_matches_a_hand_computed_result) {
  Device dev;
  CIM_EXPECT_EQ(dev.status, CIMRT_OK);

  // 4x4 identity-ish weights; out should equal the activation vector.
  const std::vector<int8_t> w = {1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1};
  const std::vector<int8_t> act = {5, -3, 100, -128};

  Buffer wb, ab, ob;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &ab.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4 * sizeof(int32_t), CIMRT_SPACE_NEAR,
                            &ob.buf),
                CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_write(wb.buf, 0, w.data(), w.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(ab.buf, 0, act.data(), act.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 0, wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 0, ab.buf, ob.buf, /*accumulate=*/false),
                CIMRT_OK);

  std::vector<int32_t> got(4, 0);
  CIM_EXPECT_EQ(cimrt_read(ob.buf, 0, got.data(), got.size() * sizeof(int32_t)),
                CIMRT_OK);
  CIM_EXPECT_EQ(got[0], 5);
  CIM_EXPECT_EQ(got[1], -3);
  CIM_EXPECT_EQ(got[2], 100);
  // -128 must survive the uint8_t storage round-trip as -128, not 128.
  CIM_EXPECT_EQ(got[3], -128);
}

CIM_TEST(cimrt_mvm_handles_negative_weights_and_activations) {
  // The weights are stored as uint8_t internally; a missing sign
  // reinterpretation would pass an all-positive test and fail here.
  Device dev;
  const std::vector<int8_t> w = {-1, 2, -3, 4,
                                  5, -6, 7, -8,
                                  -128, 127, 0, 1,
                                  1, 1, 1, 1};
  const std::vector<int8_t> act = {-7, 11, -13, 17};

  Buffer wb, ab, ob;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &ab.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4 * sizeof(int32_t), CIMRT_SPACE_NEAR,
                            &ob.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(wb.buf, 0, w.data(), w.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(ab.buf, 0, act.data(), act.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 0, wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 0, ab.buf, ob.buf, false), CIMRT_OK);

  std::vector<int32_t> got(4, 0);
  CIM_EXPECT_EQ(cimrt_read(ob.buf, 0, got.data(), got.size() * sizeof(int32_t)),
                CIMRT_OK);
  const std::vector<int32_t> want = referenceMvm(w, act, 4, 4);
  for (int i = 0; i < 4; ++i)
    CIM_EXPECT_EQ(got[i], want[i]);
}

CIM_TEST(cimrt_mvm_accumulate_chains_two_tiles) {
  // This is exactly what cim.reduce_partial expresses at the IR level: two
  // K-blocks summing into one output.
  Device dev;
  const std::vector<int8_t> w0 = {1, 1, 1, 1, 2, 2, 2, 2,
                                   3, 3, 3, 3, 4, 4, 4, 4};
  const std::vector<int8_t> w1 = {10, 0, 0, 0, 0, 10, 0, 0,
                                   0, 0, 10, 0, 0, 0, 0, 10};
  const std::vector<int8_t> act = {1, 2, 3, 4};

  Buffer w0b, w1b, ab, ob;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &w0b.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &w1b.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &ab.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4 * sizeof(int32_t), CIMRT_SPACE_NEAR,
                            &ob.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(w0b.buf, 0, w0.data(), w0.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(w1b.buf, 0, w1.data(), w1.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(ab.buf, 0, act.data(), act.size()), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_program(dev.dev, 0, w0b.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 0, ab.buf, ob.buf, /*accumulate=*/false),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 1, w1b.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 1, ab.buf, ob.buf, /*accumulate=*/true),
                CIMRT_OK);

  std::vector<int32_t> got(4, 0);
  CIM_EXPECT_EQ(cimrt_read(ob.buf, 0, got.data(), got.size() * sizeof(int32_t)),
                CIMRT_OK);

  const std::vector<int32_t> a = referenceMvm(w0, act, 4, 4);
  const std::vector<int32_t> b = referenceMvm(w1, act, 4, 4);
  for (int i = 0; i < 4; ++i)
    CIM_EXPECT_EQ(got[i], a[i] + b[i]);
}

CIM_TEST(cimrt_reprogramming_a_tile_replaces_its_weights) {
  Device dev;
  const std::vector<int8_t> first(16, 1);
  const std::vector<int8_t> second(16, 2);
  const std::vector<int8_t> act = {1, 1, 1, 1};

  Buffer wb, ab, ob;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &ab.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4 * sizeof(int32_t), CIMRT_SPACE_NEAR,
                            &ob.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(ab.buf, 0, act.data(), act.size()), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_write(wb.buf, 0, first.data(), first.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 0, wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 0, ab.buf, ob.buf, false), CIMRT_OK);
  std::vector<int32_t> got(4, 0);
  CIM_EXPECT_EQ(cimrt_read(ob.buf, 0, got.data(), 4 * sizeof(int32_t)), CIMRT_OK);
  CIM_EXPECT_EQ(got[0], 4);

  CIM_EXPECT_EQ(cimrt_write(wb.buf, 0, second.data(), second.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 0, wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 0, ab.buf, ob.buf, false), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_read(ob.buf, 0, got.data(), 4 * sizeof(int32_t)), CIMRT_OK);
  CIM_EXPECT_EQ(got[0], 8);
}

//===----------------------------------------------------------------------===//
// program / mvm error paths
//===----------------------------------------------------------------------===//

CIM_TEST(cimrt_program_rejects_bad_tiles_and_sizes) {
  Device dev;
  Buffer good, wrongSize;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &good.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 9, CIMRT_SPACE_INSITU, &wrongSize.buf),
                CIMRT_OK);

  // tiny-4x4 declares 2 tiles, so id 2 is out of range -- an invalid
  // argument, not a scheduling conflict (CIMRT_ERR_TILE_BUSY is reserved
  // for a tile that exists but was never programmed, as below).
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 2, good.buf), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 0, wrongSize.buf),
                CIMRT_ERR_SHAPE_MISMATCH);
}

CIM_TEST(cimrt_mvm_rejects_unprogrammed_tiles_and_bad_shapes) {
  Device dev;
  Buffer wb, ab, ob, shortAct, shortOut;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &ab.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4 * sizeof(int32_t), CIMRT_SPACE_NEAR,
                            &ob.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 2, CIMRT_SPACE_NEAR, &shortAct.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &shortOut.buf),
                CIMRT_OK);

  // Never programmed: computing against undefined weights would silently
  // produce a plausible-looking wrong answer.
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 1, ab.buf, ob.buf, false),
                CIMRT_ERR_TILE_BUSY);

  CIM_EXPECT_EQ(cimrt_program(dev.dev, 0, wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 0, shortAct.buf, ob.buf, false),
                CIMRT_ERR_SHAPE_MISMATCH);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 0, ab.buf, shortOut.buf, false),
                CIMRT_ERR_SHAPE_MISMATCH);
}

CIM_TEST(cimrt_rejects_null_arguments) {
  Device dev;
  cimrt_device_info info{};
  cimrt_profile profile{};
  cimrt_buffer *buf = nullptr;
  uint8_t byte = 0;

  cimrt_device *nullDev = nullptr;
  CIM_EXPECT_EQ(cimrt_open(nullptr, &nullDev), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_open(tinyTargetPath(), nullptr), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_query(nullptr, &info), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_query(dev.dev, nullptr), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_alloc(nullptr, 4, CIMRT_SPACE_NEAR, &buf),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, nullptr),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_copy(nullptr, nullptr), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_write(nullptr, 0, &byte, 1), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_read(nullptr, 0, &byte, 1), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_program(nullptr, 0, nullptr), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_mvm(nullptr, 0, nullptr, nullptr, false),
                CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_barrier(nullptr), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_profile_start(nullptr), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_profile_stop(nullptr, &profile), CIMRT_ERR_INVALID_ARG);
  CIM_EXPECT_EQ(cimrt_profile_stop(dev.dev, nullptr), CIMRT_ERR_INVALID_ARG);
}

//===----------------------------------------------------------------------===//
// Profiling
//===----------------------------------------------------------------------===//

CIM_TEST(cimrt_profile_counts_a_known_trace) {
  Device dev;
  CIM_EXPECT_EQ(cimrt_profile_start(dev.dev), CIMRT_OK);

  Buffer wb, ab, ob, qb, rb;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &ab.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4 * sizeof(int32_t), CIMRT_SPACE_NEAR,
                            &ob.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &qb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4 * sizeof(int32_t), CIMRT_SPACE_NEAR,
                            &rb.buf),
                CIMRT_OK);

  const std::vector<int8_t> w(16, 1);
  const std::vector<int8_t> act(4, 1);
  CIM_EXPECT_EQ(cimrt_write(wb.buf, 0, w.data(), w.size()), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_write(ab.buf, 0, act.data(), act.size()), CIMRT_OK);

  CIM_EXPECT_EQ(cimrt_program(dev.dev, 0, wb.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 0, ab.buf, ob.buf, false), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 0, ab.buf, ob.buf, false), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_requantize(dev.dev, ob.buf, qb.buf, 4, 32, 8, 1.0f, 0,
                                 8),
                CIMRT_OK);
  // `a` and `b` may be the same buffer -- only `out` must differ (cimrt.h)
  // -- so this reduces ob with itself, keeping the trace minimal while
  // still issuing one real cimrt_reduce_add.
  CIM_EXPECT_EQ(cimrt_reduce_add(dev.dev, rb.buf, ob.buf, ob.buf, 4, 32),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_barrier(dev.dev), CIMRT_OK);

  cimrt_profile p{};
  CIM_EXPECT_EQ(cimrt_profile_stop(dev.dev, &p), CIMRT_OK);

  // Op counts are exact and derivable from the trace above.
  CIM_EXPECT_EQ(p.programs_issued, 1u);
  CIM_EXPECT_EQ(p.mvms_issued, 2u);
  CIM_EXPECT_EQ(p.requantizes_issued, 1u);
  CIM_EXPECT_EQ(p.reduce_adds_issued, 1u);

  // bytes_transferred was permanently zero until cimrt_buffer gained a
  // device back-pointer -- recordTransfer had no way to be called. Assert a
  // lower bound rather than an exact figure: the exact total depends on how
  // a caller chooses to stage its data, and pinning it would make this test
  // break on unrelated changes without catching more bugs.
  CIM_EXPECT_GE(p.bytes_transferred, w.size() + act.size());

  // tiny-4x4 charges 10000 pJ per program, 100 pJ per mvm, 50 pJ per
  // requantize, 20 pJ per reduce_add (docs/roadmap.md's M4 cost-accounting
  // entries). With this last one every op the runtime can execute is
  // charged against the target's cost table -- there is no longer any
  // executable op that runs for free.
  CIM_EXPECT(p.estimated_energy_pj >= 10000.0 + 2 * 100.0 + 50.0 + 20.0);
  CIM_EXPECT(p.estimated_latency_ns >= 1000.0 + 2 * 10.0 + 5.0 + 2.0);
}

CIM_TEST(cimrt_status_string_covers_every_code) {
  const cimrt_status all[] = {CIMRT_OK,
                              CIMRT_ERR_NO_DEVICE,
                              CIMRT_ERR_TILE_BUSY,
                              CIMRT_ERR_SHAPE_MISMATCH,
                              CIMRT_ERR_OOM,
                              CIMRT_ERR_INVALID_ARG};
  for (cimrt_status status : all) {
    const char *text = cimrt_status_string(status);
    CIM_EXPECT(text != nullptr);
    CIM_EXPECT(std::string(text) != "unknown status");
  }
  // An out-of-range value must still return a usable string, not crash.
  CIM_EXPECT_CONTAINS(std::string(cimrt_status_string(
                          static_cast<cimrt_status>(9999))),
                      "unknown");
}

//===----------------------------------------------------------------------===//
// Lifetime, allocation failure, and profiling windows
//
// Every case below is a regression test for a defect found by reading the
// runtime rather than by a failing test -- which is the point: the suite
// above passed while all of them were live.
//===----------------------------------------------------------------------===//

CIM_TEST(cimrt_buffer_outliving_its_device_is_an_error_not_a_crash) {
  // THE ONE THAT MATTERS. cimrt_buffer holds a back-pointer to its device so
  // transfers can be costed. cimrt_close used to be a bare `delete dev`, so
  // any buffer still alive afterwards pointed at freed memory and the next
  // cimrt_write dereferenced it.
  //
  // The suite above never caught this because its RAII helpers declare
  // Device before Buffer, and C++ destroys in reverse order -- the buffer
  // always died first. Nothing about the API requires that ordering.
  cimrt_device *dev = nullptr;
  CIM_EXPECT_EQ(cimrt_open(tinyTargetPath(), &dev), CIMRT_OK);

  cimrt_buffer *buf = nullptr;
  CIM_EXPECT_EQ(cimrt_alloc(dev, 16, CIMRT_SPACE_NEAR, &buf), CIMRT_OK);

  cimrt_close(dev);

  // Orphaned: the bytes are still the buffer's own, but there is no device
  // left to charge the transfer to, and silently skipping the accounting
  // would make the cost model quietly wrong instead of loudly unavailable.
  const std::vector<uint8_t> payload(16, 0x5A);
  CIM_EXPECT_EQ(cimrt_write(buf, 0, payload.data(), payload.size()),
                CIMRT_ERR_INVALID_ARG);
  std::vector<uint8_t> readBack(16, 0);
  CIM_EXPECT_EQ(cimrt_read(buf, 0, readBack.data(), readBack.size()),
                CIMRT_ERR_INVALID_ARG);

  // Freeing an orphan must still work -- otherwise the fix for the
  // use-after-free would just turn it into a leak.
  cimrt_free(buf);
}

CIM_TEST(cimrt_freeing_a_buffer_before_its_device_leaves_no_dangling_entry) {
  // The mirror image of the test above, and the way a naive fix breaks: if
  // the device registers its buffers but cimrt_free does not deregister,
  // then cimrt_close walks a list of freed pointers. Same bug, opposite
  // direction -- ASan is what actually adjudicates this one.
  cimrt_device *dev = nullptr;
  CIM_EXPECT_EQ(cimrt_open(tinyTargetPath(), &dev), CIMRT_OK);

  cimrt_buffer *a = nullptr;
  cimrt_buffer *b = nullptr;
  CIM_EXPECT_EQ(cimrt_alloc(dev, 8, CIMRT_SPACE_NEAR, &a), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev, 8, CIMRT_SPACE_NEAR, &b), CIMRT_OK);

  cimrt_free(a);          // deregisters
  cimrt_close(dev);       // must not touch `a`
  cimrt_free(b);          // orphaned, still freeable
}

CIM_TEST(cimrt_alloc_reports_oom_rather_than_throwing_through_the_c_abi) {
  // CIMRT_ERR_OOM was declared in the ABI and returned by nothing: an
  // allocation failure escaped as a C++ exception through an extern "C"
  // boundary, which is undefined behaviour, while the header advertised a
  // status code callers could handle.
  Device dev;
  CIM_EXPECT_EQ(dev.status, CIMRT_OK);

  // SIZE_MAX, not just "far beyond any real address space": it also
  // exceeds std::vector<uint8_t>::max_size() (PTRDIFF_MAX), so
  // vector::resize's own bounds check throws std::length_error before ever
  // calling the allocator. That is deliberate, not incidental -- under
  // ASan, a request that reaches the allocator and is merely too large for
  // available memory aborts the process instead of honouring
  // allocator_may_return_null in this toolchain, which would fail this
  // test for a reason that has nothing to do with cimrt_alloc's own
  // exception handling. Never reaching the allocator sidesteps that
  // entirely while still exercising the same catch (std::length_error)
  // path cimrt_alloc must have.
  const size_t absurd = static_cast<size_t>(-1);
  cimrt_buffer *buf = nullptr;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, absurd, CIMRT_SPACE_NEAR, &buf),
                CIMRT_ERR_OOM);
  // On failure the out-parameter must be left null, not a half-built buffer.
  CIM_EXPECT(buf == nullptr);
}

CIM_TEST(cimrt_rejects_an_out_of_range_tile_as_an_invalid_argument) {
  // tiny-4x4 declares 2 tiles, so tile 7 does not exist. This used to
  // report TILE_BUSY ("tile unavailable or not programmed"), which sends a
  // caller looking for a scheduling problem that is not there.
  Device dev;
  CIM_EXPECT_EQ(dev.status, CIMRT_OK);

  Buffer weights;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &weights.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 7, weights.buf), CIMRT_ERR_INVALID_ARG);

  Buffer act, out;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4, CIMRT_SPACE_NEAR, &act.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 4 * sizeof(int32_t), CIMRT_SPACE_NEAR,
                            &out.buf),
                CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 7, act.buf, out.buf, false),
                CIMRT_ERR_INVALID_ARG);

  // A tile that exists but was never programmed is a different condition,
  // and keeps the code that actually describes it.
  CIM_EXPECT_EQ(cimrt_mvm(dev.dev, 1, act.buf, out.buf, false),
                CIMRT_ERR_TILE_BUSY);
}

CIM_TEST(cimrt_profiling_measures_a_window_not_the_whole_device_lifetime) {
  // cimrt_profile_start set a `profiling` flag that nothing ever read, and
  // stop did not reset. So counters ran from cimrt_open, start started
  // nothing, and two stops returned identical cumulative numbers -- for a
  // feature the header calls "a headline feature, not a debug tool".
  Device dev;
  CIM_EXPECT_EQ(dev.status, CIMRT_OK);

  Buffer weights;
  CIM_EXPECT_EQ(cimrt_alloc(dev.dev, 16, CIMRT_SPACE_INSITU, &weights.buf),
                CIMRT_OK);
  const std::vector<uint8_t> w(16, 1);
  CIM_EXPECT_EQ(cimrt_write(weights.buf, 0, w.data(), w.size()), CIMRT_OK);

  // Work done BEFORE the first window must not be counted in it.
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 0, weights.buf), CIMRT_OK);

  cimrt_profile first{};
  CIM_EXPECT_EQ(cimrt_profile_start(dev.dev), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_program(dev.dev, 1, weights.buf), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_profile_stop(dev.dev, &first), CIMRT_OK);
  CIM_EXPECT_EQ(first.programs_issued, 1u);

  // A second window sees only its own work, not the first window's.
  cimrt_profile second{};
  CIM_EXPECT_EQ(cimrt_profile_start(dev.dev), CIMRT_OK);
  CIM_EXPECT_EQ(cimrt_profile_stop(dev.dev, &second), CIMRT_OK);
  CIM_EXPECT_EQ(second.programs_issued, 0u);
}
