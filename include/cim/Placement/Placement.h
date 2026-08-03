//===- Placement.h - Tile placement/eviction engine ------------*- C++ -*-===//
//
// The core scheduling problem this project exists to solve (spec Sec. 3.3):
//
//   Given N physical tiles and a model with M weight sub-matrices where
//   M > N, decide which weights live where and when, to minimize total
//   reprogramming cost.
//
// Deliberately free of any MLIR/LLVM dependency. The algorithm is the
// contribution; `cim-placement` (lib/Transforms/CIMPlacement.cpp) is just
// the adapter that extracts a use-sequence from the IR, calls in here, and
// rewrites cim.program ops from the result. Keeping the two apart means the
// interesting logic can be built and tested without an LLVM toolchain.
//
//===----------------------------------------------------------------------===//
#ifndef CIM_PLACEMENT_PLACEMENT_H
#define CIM_PLACEMENT_PLACEMENT_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cim {

/// Identifies a logical weight sub-matrix (one tile-sized block of a
/// partitioned weight matrix, as emitted by `cim-partition`).
using WeightId = uint32_t;

/// Identifies a physical tile on the device.
using TileId = uint32_t;

inline constexpr TileId kNoTile = std::numeric_limits<TileId>::max();
inline constexpr WeightId kNoWeight = std::numeric_limits<WeightId>::max();
inline constexpr size_t kNeverAgain = std::numeric_limits<size_t>::max();

/// Which resident weight to evict when every tile is occupied.
enum class EvictionPolicy {
  /// Furthest-in-future (Belady/MIN). Optimal, and actually implementable
  /// here because the model graph is static: the whole use sequence is
  /// known at compile time, so we have perfect future knowledge.
  Belady,
  /// Least-recently-used. The realistic baseline a runtime cache would use;
  /// kept so benchmarks can report the win from having compile-time
  /// knowledge rather than just asserting it.
  LRU,
  /// First-in-first-out. Weakest baseline.
  FIFO,
};

const char *toString(EvictionPolicy policy);

/// A placement problem: a fixed execution order over weight sub-matrices,
/// and the number of tiles available to hold them.
struct PlacementProblem {
  /// Number of physical tiles (from the target file's `tiles.count`).
  uint32_t numTiles = 0;
  /// Weight sub-matrices in execution order. Repeats are expected and are
  /// exactly where reuse wins come from.
  std::vector<WeightId> useSequence;
  /// Human-readable label, used in reports.
  std::string name;
};

enum class ActionKind {
  /// The weight was already resident: no cim.program emitted. The win.
  Reuse,
  /// A free tile existed: cim.program emitted, nothing evicted.
  ProgramIntoFree,
  /// All tiles occupied: a resident weight was evicted to make room.
  ProgramWithEviction,
};

const char *toString(ActionKind kind);

/// What the scheduler decided for one step of the use sequence.
struct PlacementAction {
  size_t step = 0;
  WeightId weight = kNoWeight;
  TileId tile = kNoTile;
  ActionKind kind = ActionKind::Reuse;
  /// Only meaningful for ProgramWithEviction.
  WeightId evicted = kNoWeight;
};

struct PlacementResult {
  std::vector<PlacementAction> actions;
  /// Number of cim.program ops the pass would emit. This is the quantity
  /// the whole optimizer exists to minimize.
  uint64_t programs = 0;
  /// Steps satisfied from an already-resident tile.
  uint64_t reuses = 0;
  uint64_t evictions = 0;
  /// Distinct weight sub-matrices seen. When `programs == distinctWeights`,
  /// every weight was programmed exactly once — on non-volatile hardware
  /// that means programmed at install time and never again.
  uint64_t distinctWeights = 0;
  EvictionPolicy policy = EvictionPolicy::Belady;
};

/// Run the scheduler. Returns an empty result for an empty use sequence.
/// If `numTiles` is 0 the problem is unsatisfiable and the result is empty
/// with `programs == 0`; callers should treat that as an error.
PlacementResult computePlacement(const PlacementProblem &problem,
                                  EvictionPolicy policy);

/// Replay the schedule against a simulated tile array and confirm that at
/// every step the named tile really does hold the weight being used. A
/// performance number without a correctness check next to it is worthless
/// (spec Sec. 10), and this is that check for the placement pass.
/// Returns true on success; on failure writes a description to `error`.
bool validatePlacement(const PlacementProblem &problem,
                        const PlacementResult &result, std::string *error);

} // namespace cim

#endif // CIM_PLACEMENT_PLACEMENT_H
