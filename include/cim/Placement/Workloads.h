//===- Workloads.h - The v0.1 benchmark workloads --------------*- C++ -*-===//
//
// The spec Sec. 10 workload table, expressed as placement problems. These
// describe weight *residency pressure* only — the shapes below determine
// how many tile-sized weight blocks exist and in what order they are used,
// which is everything the placement pass reasons about. Numerical
// correctness of the arithmetic is the functional simulator's job
// (runtime/src/simulator), not this layer's.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_PLACEMENT_WORKLOADS_H
#define CIM_PLACEMENT_WORKLOADS_H

#include "cim/Placement/Placement.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cim {

/// A benchmark workload: a placement problem plus the metadata needed to
/// split its use sequence into per-inference windows for cost reporting.
struct Workload {
  std::string name;
  std::string probes;
  PlacementProblem problem;
  size_t stepsPerInference = 0;
  uint32_t inferences = 1;
};

/// Number of tile-sized blocks a [K x N] weight matrix partitions into on a
/// target with [rows x cols] tiles, matching `cim-partition` (spec Sec. 6):
///   tiles_k = ceil(K / rows), tiles_n = ceil(N / cols)
/// Ragged edges count as full blocks because they are zero-padded.
uint32_t partitionBlockCount(uint32_t k, uint32_t n, uint32_t tileRows,
                              uint32_t tileCols);

/// One matmul layer repeated `inferences` times: blocks are swept in a
/// fixed order, which is the execution order `cim-partition` emits.
Workload makeMatmulWorkload(const std::string &name, const std::string &probes,
                             uint32_t blocks, uint32_t numTiles,
                             uint32_t inferences);

/// Several sequential matmul layers (an MLP or a transformer FFN block),
/// each with its own block count, all swept once per inference.
Workload makeLayeredWorkload(const std::string &name, const std::string &probes,
                              const std::vector<uint32_t> &blocksPerLayer,
                              uint32_t numTiles, uint32_t inferences);

/// The five v0.1 workloads from spec Sec. 10, sized against a target with
/// `numTiles` tiles of [tileRows x tileCols].
std::vector<Workload> makeV01Workloads(uint32_t numTiles, uint32_t tileRows,
                                        uint32_t tileCols, uint32_t inferences);

} // namespace cim

#endif // CIM_PLACEMENT_WORKLOADS_H
