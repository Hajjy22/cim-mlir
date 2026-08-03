//===- Workloads.cpp - The v0.1 benchmark workloads ------------*- C++ -*-===//

#include "cim/Placement/Workloads.h"

namespace cim {

uint32_t partitionBlockCount(uint32_t k, uint32_t n, uint32_t tileRows,
                              uint32_t tileCols) {
  if (tileRows == 0 || tileCols == 0)
    return 0;
  const uint32_t tilesK = (k + tileRows - 1) / tileRows;
  const uint32_t tilesN = (n + tileCols - 1) / tileCols;
  return tilesK * tilesN;
}

Workload makeMatmulWorkload(const std::string &name, const std::string &probes,
                             uint32_t blocks, uint32_t numTiles,
                             uint32_t inferences) {
  Workload wl;
  wl.name = name;
  wl.probes = probes;
  wl.problem.name = name;
  wl.problem.numTiles = numTiles;
  wl.stepsPerInference = blocks;
  wl.inferences = inferences;

  wl.problem.useSequence.reserve(static_cast<size_t>(blocks) * inferences);
  for (uint32_t rep = 0; rep < inferences; ++rep)
    for (uint32_t b = 0; b < blocks; ++b)
      wl.problem.useSequence.push_back(b);

  return wl;
}

Workload makeLayeredWorkload(const std::string &name, const std::string &probes,
                              const std::vector<uint32_t> &blocksPerLayer,
                              uint32_t numTiles, uint32_t inferences) {
  Workload wl;
  wl.name = name;
  wl.probes = probes;
  wl.problem.name = name;
  wl.problem.numTiles = numTiles;
  wl.inferences = inferences;

  // Weight IDs are globally unique across layers: two layers never share a
  // weight sub-matrix, so residency competition between them is real.
  std::vector<WeightId> oneInference;
  WeightId next = 0;
  for (uint32_t blocks : blocksPerLayer)
    for (uint32_t b = 0; b < blocks; ++b)
      oneInference.push_back(next++);

  wl.stepsPerInference = oneInference.size();
  wl.problem.useSequence.reserve(oneInference.size() * inferences);
  for (uint32_t rep = 0; rep < inferences; ++rep)
    wl.problem.useSequence.insert(wl.problem.useSequence.end(),
                                  oneInference.begin(), oneInference.end());

  return wl;
}

std::vector<Workload> makeV01Workloads(uint32_t numTiles, uint32_t tileRows,
                                        uint32_t tileCols,
                                        uint32_t inferences) {
  std::vector<Workload> workloads;

  // mm-fit: weights fit entirely in tiles. Best case — zero reprogramming
  // after install. Sized so the partition exactly fills the device: this is
  // the spec Sec. 17 worked example ([512 x 1024] on 8 tiles of 256x256).
  {
    const uint32_t blocks =
        partitionBlockCount(tileRows * 2, tileCols * (numTiles / 2), tileRows, tileCols);
    workloads.push_back(makeMatmulWorkload(
        "mm-fit", "best case: zero reprogramming after install", blocks,
        numTiles, inferences));
  }

  // mm-spill-2x: weights 2x tile capacity. Reprogramming pressure, and the
  // first case where eviction policy quality shows up.
  workloads.push_back(makeMatmulWorkload(
      "mm-spill-2x", "reprogramming pressure, eviction policy quality",
      numTiles * 2, numTiles, inferences));

  // mm-spill-8x: weights 8x tile capacity. Thrashing behavior.
  workloads.push_back(makeMatmulWorkload(
      "mm-spill-8x", "thrashing behavior", numTiles * 8, numTiles, inferences));

  // mlp-3layer: 3 sequential matmuls. Cross-layer residency scheduling.
  workloads.push_back(makeLayeredWorkload(
      "mlp-3layer", "cross-layer residency scheduling",
      {numTiles / 2, numTiles / 2, numTiles / 2}, numTiles, inferences));

  // bert-ffn: one transformer FFN block (2 matmuls, 4x expand then project).
  // A shape people actually care about.
  {
    const uint32_t hidden = tileCols * 2;
    const uint32_t ffn = hidden * 4;
    const uint32_t up = partitionBlockCount(hidden, ffn, tileRows, tileCols);
    const uint32_t down = partitionBlockCount(ffn, hidden, tileRows, tileCols);
    workloads.push_back(makeLayeredWorkload(
        "bert-ffn", "a shape people actually care about", {up, down}, numTiles,
        inferences));
  }

  return workloads;
}

} // namespace cim
