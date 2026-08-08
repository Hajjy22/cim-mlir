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

/// A steady-state placement problem: one loop iteration's use sequence,
/// solved for a schedule that stays valid under arbitrary repetition --
/// see computeSteadyStatePlacement's own comment for what that means and
/// why it is a different question from PlacementProblem/computePlacement,
/// which only ever sees one pass over its use sequence.
struct SteadyStateProblem {
  /// Number of physical tiles (from the target file's `tiles.count`).
  uint32_t numTiles = 0;
  /// One loop iteration's use sequence, in program order. Every entry is
  /// necessarily loop-invariant -- the same physical weight sub-matrix on
  /// every iteration -- by construction of how a caller recovers this
  /// from real IR (`lib/Transforms/CIMPlacement.cpp`'s `weightIdentity`
  /// refuses a weight with a dynamic offset outright, and cim-partition's
  /// own tiling never produces one: the loop induction variable only
  /// ever appears in an activation- or output-side subview, not a
  /// weight's).
  std::vector<WeightId> bodySequence;
  /// Human-readable label, used in reports.
  std::string name;
};

/// What computeSteadyStatePlacement decided for one iteration.
struct SteadyStateResult {
  /// One action per step of `bodySequence`, in the same order -- the
  /// same shape as PlacementResult::actions, and meant to be rewritten
  /// by exactly the same caller logic: a Reuse means erase this step's
  /// cim.program and rewire its consumers to the earlier one; anything
  /// else means keep it and assign it the named tile.
  std::vector<PlacementAction> body;
  /// Tile ids `[0, pinnedTileCount)` named in `body` are safe to hoist to
  /// permanent, cross-iteration residency: installed once, before the
  /// loop, and never reprogrammed again. Tile ids
  /// `[pinnedTileCount, numTiles)` are "stream" tiles, whose content
  /// changes within a single iteration -- mirroring the rule
  /// `lib/Transforms/CIMPlacement.cpp`'s own accidental hoisting already
  /// followed (a tile written more than once per iteration cannot be
  /// assumed stable across iterations either), nothing programmed into
  /// one of these may be hoisted.
  uint32_t pinnedTileCount = 0;
  /// This body's own program count: what a real `cim.program` fires once
  /// PER ITERATION, as opposed to a pinned tile's install, which fires
  /// once total no matter how many iterations run.
  uint64_t programsPerIteration = 0;
  /// False for an infeasible problem (an empty `bodySequence`, or zero
  /// tiles); callers must not act on an infeasible result.
  bool feasible = false;
};

/// Solve for a placement that stays valid across arbitrarily many
/// repetitions of `problem.bodySequence` -- pin-and-stream: the
/// `numTiles - 1` (at most) weights used most often within the body are
/// held permanently resident, and every other weight streams through the
/// one remaining tile, reprogrammed whenever it changes.
///
/// This is the DELIBERATE version of what `lib/Transforms/CIMPlacement
/// .cpp`'s own hoisting used to reach only as an accidental consequence
/// of Belady's own victim-selection tie-break (see that file's decision
/// 4 for the full account of why that was fragile). Provably optimal
/// when every weight in `bodySequence` is used exactly once -- the shape
/// cim-partition itself emits, one cim.program per weight block per
/// matmul: pinning the `numTiles - 1` most-used weights and streaming
/// the remainder through the last tile achieves the proven lower bound
/// `distinctWeights - (numTiles - 1)` programs per iteration, because a
/// weight nothing in the body ever reprograms holds its tile permanently
/// and at most `numTiles - 1` weights can do that at once.
///
/// When a weight is used MORE than once within one body, pin-and-stream
/// is a correct heuristic rather than a proven optimum -- the exact
/// answer there is a residency fixed-point problem over the whole body
/// (letting the one streaming tile hold a weight across a gap between
/// two of its own uses can occasionally beat evicting it for something
/// used exactly once in between), which this does not attempt to solve
/// exactly. The caller is expected to compare this against its own
/// existing schedule and keep whichever costs less, so a case where the
/// heuristic falls short is never worse than not calling this at all.
SteadyStateResult computeSteadyStatePlacement(const SteadyStateProblem &problem);

/// Confirms `result` really is a fixed point, using nothing but the
/// existing, already-trusted `validatePlacement` -- not a second,
/// separately-trusted checker. Builds one ordinary, flattened
/// PlacementProblem/PlacementResult: a bootstrap installing each pinned
/// tile's real starting content plus (if streaming is used) the stream
/// tile's own first-ever content, followed by THREE repetitions of
/// `result.body` back to back -- the first using `body`'s own actions
/// verbatim (valid starting from the bootstrap), the second and third
/// substituting freshly recomputed labels for the stream tile's steps
/// only (a pinned tile's Reuse claim needs no relabeling: once installed
/// its content never changes again, so the same claim stays true
/// forever). A bug that makes the SECOND repetition's claims false --
/// the entire content of "steady state" -- is caught by
/// `validatePlacement`'s own replay exactly the way any other invalid
/// schedule would be.
bool validateSteadyStatePlacement(const SteadyStateProblem &problem,
                                   const SteadyStateResult &result,
                                   std::string *error);

} // namespace cim

#endif // CIM_PLACEMENT_PLACEMENT_H
