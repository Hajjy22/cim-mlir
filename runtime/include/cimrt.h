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

typedef enum {
  CIMRT_OK = 0,
  CIMRT_ERR_NO_DEVICE,
  CIMRT_ERR_TILE_BUSY,
  CIMRT_ERR_SHAPE_MISMATCH,
  CIMRT_ERR_OOM,
} cimrt_status;

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
 * simulated hardware rather than the static analytical model. */
typedef struct cimrt_profile {
  uint64_t programs_issued;
  uint64_t mvms_issued;
  uint64_t bytes_transferred;
  double estimated_energy_pj;
  double estimated_latency_ns;
} cimrt_profile;

/* --- device --- */
cimrt_status cimrt_open(const char *target_name, cimrt_device **out);
void cimrt_close(cimrt_device *dev);
cimrt_status cimrt_query(cimrt_device *dev, cimrt_device_info *out);

/* --- memory --- */
cimrt_status cimrt_alloc(cimrt_device *dev, size_t bytes, cimrt_space space,
                          cimrt_buffer **out);
cimrt_status cimrt_copy(cimrt_buffer *dst, const cimrt_buffer *src);
void cimrt_free(cimrt_buffer *buf);

/* --- the two operations that matter (spec Sec. 3.3) --- */
cimrt_status cimrt_program(cimrt_device *dev, cimrt_tile_id tile,
                            const cimrt_buffer *weights);
cimrt_status cimrt_mvm(cimrt_device *dev, cimrt_tile_id tile,
                        const cimrt_buffer *act, cimrt_buffer *out,
                        bool accumulate);

/* --- sync --- */
cimrt_status cimrt_barrier(cimrt_device *dev);

/* --- profiling: mandatory, not optional --- */
cimrt_status cimrt_profile_start(cimrt_device *dev);
cimrt_status cimrt_profile_stop(cimrt_device *dev, cimrt_profile *out);

#ifdef __cplusplus
}
#endif

#endif /* CIMRT_H */
