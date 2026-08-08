/*===- cimrt.h - CIM runtime C ABI (spec Sec. 8) -----------------------====*
 *
 * Deliberately tiny. Plain C ABI so anything can bind to it. Mirrors the
 * `cim` dialect ops roughly 1:1 (cim.program -> cimrt_program, cim.mvm ->
 * cimrt_mvm, ...) since cim-lower-to-target (spec Sec. 6, Pass 7) lowers
 * straight to these calls in v0.1.
 *
 * Design note on profiling (spec Sec. 8): most vendor CIM stacks make
 * profiling an afterthought, which is a direct cause of "this hardware is
 * unusable." Counters for programs issued, MVMs issued, bytes transferred,
 * and estimated energy are first-class here from day one — the profiler is
 * a headline feature, not a debug tool.
 *===----------------------------------------------------------------------*/
#ifndef CIMRT_H
#define CIMRT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cimrt_device cimrt_device;
typedef struct cimrt_buffer cimrt_buffer;
typedef uint32_t cimrt_tile_id;

/* Appended to, never reordered: these values are part of the ABI. */
typedef enum {
  CIMRT_OK = 0,
  CIMRT_ERR_NO_DEVICE,
  CIMRT_ERR_TILE_BUSY,
  CIMRT_ERR_SHAPE_MISMATCH,
  CIMRT_ERR_OOM,
  CIMRT_ERR_INVALID_ARG,   /* null pointer, or an out-of-range offset/size */
} cimrt_status;

/* Human-readable form of a status, for diagnostics. Never returns NULL. */
const char *cimrt_status_string(cimrt_status status);

/* Mirrors #cim.space<host|near|insitu> (spec Sec. 3.4). */
typedef enum {
  CIMRT_SPACE_HOST = 0,
  CIMRT_SPACE_NEAR,
  CIMRT_SPACE_INSITU,
} cimrt_space;

/* Result of cimrt_query. Reports the geometry/precision of the opened
 * target so callers (and cim-lower-to-target) don't need to re-parse the
 * target YAML at runtime. */
typedef struct cimrt_device_info {
  char name[64];
  uint32_t num_tiles;
  uint32_t tile_rows;
  uint32_t tile_cols;
  bool persistent; /* true on non-volatile targets, spec Sec. 3.1 */
} cimrt_device_info;

/* Populated by cimrt_profile_stop. Spec Sec. 8: profiling is mandatory,
 * not optional -- this struct is what backs the cost reports in
 * cim-cost-report (spec Sec. 6, Pass 8) when running against real or
 * simulated hardware rather than the static analytical model.
 *
 * Accounting rule, so callers and tests are not guessing:
 *   bytes_transferred counts every byte moved across the runtime's buffer
 *   boundary by cimrt_copy, cimrt_write and cimrt_read. cimrt_program and
 *   cimrt_mvm do NOT record transfers -- their cost is whatever
 *   costs.program / costs.mvm in the target file says it is. */
typedef struct cimrt_profile {
  uint64_t programs_issued;
  uint64_t mvms_issued;
  uint64_t bytes_transferred;
  double estimated_energy_pj;
  double estimated_latency_ns;
} cimrt_profile;

/* --- device ---
 *
 * target_name is either a bare target name, resolved to
 * targets/<name>.yaml relative to the working directory, or -- if it
 * contains '/' or ends in ".yaml" -- a path to a target file, so callers
 * that are not run from the repo root can still open a device.
 *
 * TRUST BOUNDARY: target_name reaches here from cim.device_open's `target`
 * attribute, which today only ever comes from IR this project's own passes
 * emit. If anything ever compiles IR from an untrusted source, this is an
 * unconstrained filesystem path read -- constrain it to a fixed target
 * search directory (reject '..' components, refuse absolute paths outside
 * it) before that changes. Not done now because there is no untrusted
 * caller yet and doing it early would be speculative.
 *
 * A name ending in "-hw" routes to the real hardware backend rather than
 * the functional simulator. */
cimrt_status cimrt_open(const char *target_name, cimrt_device **out);

/* Closes the device and orphans every cimrt_buffer still allocated from it:
 * such a buffer's own bytes remain valid and cimrt_free still works on it,
 * but any call that would need the device -- cimrt_write/cimrt_read/
 * cimrt_copy's cost accounting -- returns CIMRT_ERR_INVALID_ARG instead of
 * touching freed memory. Silently skipping the accounting was considered
 * and rejected: a cost model that quietly stops counting is worse than one
 * that visibly refuses. */
void cimrt_close(cimrt_device *dev);
cimrt_status cimrt_query(cimrt_device *dev, cimrt_device_info *out);

/* --- memory --- */
cimrt_status cimrt_alloc(cimrt_device *dev, size_t bytes, cimrt_space space,
                          cimrt_buffer **out);
cimrt_status cimrt_copy(cimrt_buffer *dst, const cimrt_buffer *src);
/* Safe to call on a buffer whose device has already been closed. */
void cimrt_free(cimrt_buffer *buf);

/* Copies `bytes` starting at `src_offset` in `src` to `bytes` starting at
 * `dst_offset` in `dst` -- a byte-range generalization of cimrt_copy (which
 * requires the two buffers be the same whole size) for cim-lower-to-target's
 * benefit: materializing a genuine (non-identity) memref.subview of a
 * device-space buffer into a fresh buffer of the slice's own size, the one
 * case cimrt_copy's whole-buffer-only contract cannot express. `dst` and
 * `src` must be different buffers (no aliasing support, matching
 * cimrt_reduce_add's own out/input restriction); `dst_offset + bytes` must
 * not exceed `dst`'s size, nor `src_offset + bytes` exceed `src`'s. */
cimrt_status cimrt_copy_range(cimrt_buffer *dst, size_t dst_offset,
                               const cimrt_buffer *src, size_t src_offset,
                               size_t bytes);

/* Host <-> buffer transfer.
 *
 * Without these there is no way to get a nonzero byte into a buffer at all:
 * cimrt_alloc zero-fills and cimrt_copy only moves buffer-to-buffer, so no
 * caller could ever observe a nonzero cimrt_mvm result.
 *
 * Deliberately copy-in/copy-out rather than a cimrt_map() handing back a
 * raw pointer: an unobserved write through a mapped pointer would be a
 * transfer the cost model never sees, and "every transfer is explicit"
 * (spec Sec. 3.4) is the property the whole cost model rests on. */
cimrt_status cimrt_write(cimrt_buffer *dst, size_t dst_offset,
                          const void *src, size_t bytes);
cimrt_status cimrt_read(const cimrt_buffer *src, size_t src_offset,
                         void *dst, size_t bytes);

/* --- the two operations that matter (spec Sec. 3.3) --- */
cimrt_status cimrt_program(cimrt_device *dev, cimrt_tile_id tile,
                            const cimrt_buffer *weights);
cimrt_status cimrt_mvm(cimrt_device *dev, cimrt_tile_id tile,
                        const cimrt_buffer *act, cimrt_buffer *out,
                        bool accumulate);

/* Requantize (spec Sec. 5.3): models a digital requantizer or an analog
 * ADC readout. Reads `count` elements of `in_bits`-wide signed integers
 * from `input`, computes `quantized = zero_point + round(value / scale)`
 * (round-half-away-from-zero), clamps to the SIGNED range `effective_bits`
 * can hold, and writes `count` elements of `out_bits`-wide signed integers
 * to `output`.
 *
 * `in_bits`/`out_bits` are explicit, not inferred, for the same reason
 * every other call here takes an explicit byte count: cimrt_buffer carries
 * no dtype of its own. `in_bits` and `out_bits` may differ (narrowing,
 * e.g. i32->i8, is the common case; widening is also valid) and each must
 * be a positive multiple of 8 -- this ABI is exactly as byte-addressed as
 * everywhere else in this project. `effective_bits` must be positive and
 * not exceed `out_bits`.
 *
 * Not yet counted by cimrt_profile_stop: the target schema's `costs:`
 * section has no requantize/readout entry (spec target-format.md), so this
 * call currently contributes nothing to programs_issued/mvms_issued/
 * energy -- a known simplification (spec M4), not a silent omission. */
cimrt_status cimrt_requantize(cimrt_device *dev, const cimrt_buffer *input,
                               cimrt_buffer *output, size_t count,
                               uint32_t in_bits, uint32_t out_bits,
                               float scale, int32_t zero_point,
                               uint32_t effective_bits);

/* Elementwise wrapping addition of two device-space buffers into a third:
 * out[i] = a[i] + b[i] (mod 2^bits), for `count` signed integers of
 * `bits`-wide elements each, in `a`, `b`, and `out` alike. Wrapping, not
 * saturating -- a real fixed-width hardware accumulator wraps, and this
 * must agree with Interpreter.cpp's runReducePartial, which computes the
 * identical reduction host-side over the interpreter's own buffers.
 *
 * cim.reduce_partial's job (spec Sec. 5.4 rule 4) when a logical weight
 * matrix spans multiple K-tiles and their partial sums must be summed into
 * one: an N-operand reduce_partial lowers to N-1 chained calls here, one
 * per pairwise add, matching the interpreter's own left-to-right fold.
 * `out` must not alias `a` or `b`. `bits` must be a positive multiple of
 * 8, matching every other width parameter in this ABI -- always 32 in
 * today's real pipeline (cim.mvm's accumulator is i32), but not hardcoded
 * to it, the same reason cimrt_requantize takes in_bits/out_bits rather
 * than assuming a width.
 *
 * Not yet counted by cimrt_profile_stop, matching cimrt_requantize's own
 * note: the target schema has no reduce/accumulate cost entry (spec M4).
 */
cimrt_status cimrt_reduce_add(cimrt_device *dev, cimrt_buffer *out,
                               const cimrt_buffer *a, const cimrt_buffer *b,
                               size_t count, uint32_t bits);

/* --- sync --- */
cimrt_status cimrt_barrier(cimrt_device *dev);

/* --- profiling: mandatory, not optional ---
 *
 * cimrt_profile_start opens a fresh measurement window: it snapshots the
 * device's running counters, and cimrt_profile_stop reports the DELTA since
 * that snapshot, not the cumulative totals since cimrt_open. Two windows
 * around different work report different numbers; a stop with no matching
 * start reports zero rather than whatever accumulated since open, since
 * "no window was opened" and "an empty window" should not be
 * indistinguishable from "everything since the device was created". */
cimrt_status cimrt_profile_start(cimrt_device *dev);
cimrt_status cimrt_profile_stop(cimrt_device *dev, cimrt_profile *out);

#ifdef __cplusplus
}
#endif

#endif /* CIMRT_H */
