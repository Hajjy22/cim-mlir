# Roadmap

Milestones from the v0.1 spec (Section 13). Each has a public artifact —
nothing counts until it is public.

**Current state.** M0, M1 and M3 are done. Four of the eight lowering passes are
real: `cim-detect` annotates eligible matmuls, `cim-partition` lowers them into
per-tile `cim.program`/`cim.mvm` with partial-sum reduction and explicit space
transfers, `cim-placement` rewrites that IR from a Belady schedule --
eliminating redundant weight programming, assigning tile ids from the
solution rather than round-robin, and hoisting loop-invariant `cim.program`
ops out of an `scf.for` when doing so is provably safe -- and `cim-cost-report`
(Pass 8) walks the final placed IR and emits the project's publishable numbers,
reusing `cim-bench`'s own `CostReport`/JSON format rather than a second cost
path (see the M3 entry below for how it accounts for loop trip counts).

Separately, a full read-through of `cimrt` and the interpreter turned up seven
real defects -- a use-after-free when a `cimrt_buffer` outlived its device, an
allocation failure that threw a C++ exception across the `extern "C"`
boundary instead of returning `CIMRT_ERR_OOM`, misleading `CIMRT_ERR_TILE_BUSY`
on an out-of-range tile id, a profiling window that never actually windowed
anything, an `erbium-hw` open path that could never succeed, an undocumented
filesystem trust boundary, and a sub-byte element type (`i1`/`i4`) silently
computing zero-byte allocations via `bitWidth / 8` truncating to zero. All
seven are fixed, each with a regression test that failed against the unfixed
code first and, for the two memory-safety ones, a mutation test confirming
ASan actually catches the bug when the fix is reverted
(`test/unit/cimrt_test.cpp`, `test/Run/errors.mlir`).

What "provably safe" means, precisely, because it is not the same problem
`cim-bench`'s simulator solves: a `cim.program` is hoisted only when its
physical tile is written by no other `cim.program` within one textual loop
iteration, and only out of a loop whose trip count is a compile-time
constant proven positive. This exactly reproduces the headline `mm-fit`
result on real IR -- a model whose weights entirely fit in tiles reprograms
once, however many inferences run. It does **not** reproduce `cim-bench`'s
`mm-spill-*` numbers: those come from Belady solved over the *whole*
flattened N-inference sequence, which can find reuse this pass's
single-iteration, tile-local check cannot see. Under spill, hoisting still
finds and moves whatever subset of tiles genuinely stays stable for a full
iteration -- see `placement_partially_hoists_when_only_some_tiles_are_stable`
in `test/mlir/pipeline_e2e_test.cpp` -- but the spill workloads in
`bench/workloads/README.md` still describe what the standalone simulator
computes, not what this pass emits. A full N-inference Belady solve on
compiled IR, if it is ever wanted, is separate work from what landed here.

## M0 — Environment and orientation
- [x] Repository scaffold matching this layout.
- [x] `cim-opt` compiles and runs against MLIR 18. Note: no from-source LLVM
  build is needed — Debian/Ubuntu ship `libmlir-18-dev`, which is what CI
  installs. That turned a 45–90 minute cold CI build into an apt install.
- [ ] Toy tutorial chapters 1–7 completed.
- [ ] Erbium emulator running locally.

## M1 — Dialect skeleton (complete)
- [x] `cim` dialect defined in ODS: types, attributes, all nine ops
  (`include/cim/Dialect/`), compiling and parsing.
- [x] Verifiers implemented for rules 1 and 4 plus shape/geometry/accumulator
  checks (`lib/Dialect/CIM/IR/CIMOps.cpp`). Rules 2, 3, and 5 need
  information a single op cannot see and are documented as such in the
  source rather than silently skipped.
- [x] FileCheck round-trip test per op, plus `invalid.mlir` proving the
  verifiers actually reject bad IR (10 tests, all passing).
- [x] CI builds and tests both layers.

## M2 — Functional correctness (in progress)
- [x] Pass 1 `cim-detect`: annotates INT8 matmuls with a constant weight
  operand. Rejects f32, activation-times-activation, narrow accumulators,
  and convolution (`test/Transforms/cim-detect.mlir`).
- [x] Pass 2 `cim-partition`: lowers a candidate into per-tile
  `cim.tile_alloc`/`cim.program`/`cim.mvm`, a `cim.reduce_partial` over the
  contraction dimension, and explicit `cim.copy` staging into near memory.
  Tile geometry and per-op costs come from `-target-yaml`, never from a
  hardcoded constant.
- [x] Functional simulator + `cimrt` runtime with a real INT8 MVM reference
  implementation (`runtime/src/simulator/simulator.cpp`).
- [ ] Passes 5 and 7 (`cim-insert-transfers`, `cim-lower-to-target`) are
  still stubs. `cim-partition` currently emits its own transfers, so
  pass 5 has little to do until multi-layer models arrive.
- [ ] End-to-end: an ONNX INT8 matmul compiles and produces numerically
  correct output vs. PyTorch. Nothing yet connects the compiled IR to the
  simulator, so there is no numerical check across the whole pipeline.

### Known limits of cim-partition
Each is refused with a warning and the `linalg` op left intact, so the
module stays correct and is simply not offloaded:
- Only `linalg.matmul_transpose_b` (weights `[N x K]`, matching `cim.mvm`'s
  output-major convention). A plain `linalg.matmul` needs a transpose first.
- Only a single output row: `cim.mvm` is a matrix-vector primitive and the
  v0.1 contract is matrix-vector.
- Only exact multiples of the tile geometry. Spec Sec. 6 calls for
  zero-padding ragged edges; that needs a pad-and-copy sequence.

## M3 — The placement pass
- [x] Belady/MIN eviction implemented and unit-tested, with LRU and FIFO
  baselines to compare against (`lib/Placement/`, `test/unit/placement_test.cpp`).
  Every schedule is replayed through `validatePlacement()` before its
  numbers are trusted.
- [x] Analytical cost model over the target file's cost table, including
  install-vs-steady-state breakdown and amortization
  (`lib/Placement/CostReport.cpp`).
- [x] `cim-bench` running the five v0.1 workloads end to end, emitting
  results JSON with target-file hash, git commit, and date
  (`bench/workloads/README.md` has the current numbers).
- [x] Plot script in the repo (`bench/plots/plot_residency.py`).
- [x] `cim-placement` rewriting actual IR. It recovers the use sequence from
  the `cim.program` ops in each block, solves with Belady, and rewrites:
  redundant programs are erased and their `cim.mvm` consumers rewired to the
  resident already in the tile, and surviving programs get their tile id from
  the schedule. Verified numerically — every case runs both with and without
  the pass and the outputs must be identical (`test/mlir/pipeline_e2e_test.cpp`,
  `test/python/test_numerical_differential.py`), because placement is an
  optimization and is not allowed to change an answer.

  The identity of a weight sub-matrix is `(root allocation, byte offset,
  shape)`, not the SSA value: `cim-partition` emits a fresh `memref.subview`
  per block per matmul, so keying on the value would make every weight
  distinct and the pass would find no reuse while appearing to work.
- [x] Hoisting `cim.program` out of an `scf.for` when it is the sole writer
  of its physical tile within one loop iteration, and the loop's trip count
  is a compile-time constant proven positive (never out of a loop that
  might run zero times, and never out of a loop nested inside another --
  v0.1 handles one level). `lib/Interpreter/Interpreter.cpp` gained real
  `scf.for` execution (bound induction variable, static bounds only, no
  `iter_args`) to make this checkable numerically rather than only
  structurally -- see `test/mlir/pipeline_e2e_test.cpp`'s loop-hoisting
  cases, `test/Transforms/cim-placement-loop.mlir` for the IR shape, and
  `test/Run/placement-loop.mlir` for the `cimrt_profile` counters actually
  moving. This is the `mm-fit` claim (fits entirely -> reprograms once) on
  compiled IR; it is not a full N-inference Belady solve -- see the note
  above the milestone list.
- [ ] A full N-inference Belady solve on compiled IR, matching what
  `cim-bench`'s simulator computes for the spill workloads. The current
  hoist is deliberately more conservative (see above); closing this gap
  means reasoning about the whole flattened use sequence across iterations,
  not just whether one tile is stable within one.
- [x] `cim-cost-report` (Pass 8): walks the final, already-placed IR and
  emits a JSON cost report, reusing `cim::CostReport`/`toJson`
  (`lib/Placement/CostReport.cpp`) rather than a second cost model that
  could drift from `cim-bench`'s. Every `cim.program`/`cim.mvm` is weighted
  by the product of its enclosing loops' compile-time-constant trip counts
  (`cim::getConstantTripCount`, `lib/Transforms/LoopAnalysis.cpp`, shared
  with `cim-placement`'s hoisting check) before being counted: a hoisted
  `cim.program` executes once regardless of where it sits textually, one
  left inside an `scf.for` executes trip-count times, and a plain
  walk-and-count would misreport exactly the IR loop hoisting now produces.
  An op under a loop whose trip count is not a compile-time constant is
  excluded from the totals and flagged (`trip-count-complete: false` in the
  JSON header, plus a diagnostic) rather than assumed to fire once.
  Verified against real execution, not just against itself:
  `test/mlir/cost_report_e2e_test.cpp` asserts the predicted
  `programs`/`mvms` equal what `cim-run --profile` actually reports from
  `cimrt`'s own counters, for straight-line, spill, and both hoisted and
  unhoisted loop cases -- and a mutation test (disabling the trip-count
  weighting) confirms the loop cases actually fail without it.
- [ ] Volatile-vs-non-volatile comparison plot showing that persistence
  changes the optimum, not just the magnitude.

## M4 — Second target and generalization (future)
- `targets/generic-digital-cim.yaml` already exists as a placeholder
  second-class target file; needs real (or better-estimated) numbers and a
  working lowering to prove retargetability.
- `cim-legalize-precision` with real `effective_bits` modeling.

## M5 — Community and real hardware (future)
- Real Erbium-8T hardware backend (`runtime/src/erbium/erbium_backend.cpp`
  currently stubs every entry point with `CIMRT_ERR_NO_DEVICE`).
- Upstream contribution to AiNEKKO/Erbium.
- Conference talk submission.

## M6 — Decision point (future)
Assess against the signals in spec Section 15 (external engagement, MLIR
learning curve, competitive landscape) and choose: raise/recruit, take a
role at a company in the space, or continue as an open-source maintainer.

## Deviations from the spec

Two places where the spec as written could not be implemented literally.
Both were found by compiling it, not by reading it.

**Pass naming.** The spec names the seventh pass `--cim-lower-to-<target>`
(implying one flag per target). MLIR pass registration requires a single
fixed pass name, so this is one `cim-lower-to-target` pass parameterized by
a `-target-yaml=<path>` option instead (`include/cim/Transforms/Passes.td`).

**`cim.tile_alloc` syntax.** The spec writes it as
`cim.tile_alloc %dev {id = 0} : !cim.tile<256x256xi8>`, printing only the
result type. That is unparseable: `!cim.device` carries a target-name
parameter, so the operand type cannot be inferred and must appear in the
syntax. It prints as a functional type instead:
`%t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"erbium-8t">) -> !cim.tile<256x256xi8>`.
