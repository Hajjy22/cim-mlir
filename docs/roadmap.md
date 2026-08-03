# Roadmap

Milestones from the v0.1 spec (Section 13). Each has a public artifact —
nothing counts until it is public.

**Current state.** M0 and M1 are done. The first two lowering passes are
real: `cim-detect` annotates eligible matmuls and `cim-partition` lowers
them into per-tile `cim.program`/`cim.mvm` with partial-sum reduction and
explicit space transfers, driven by the target file's tile geometry. M3's
substance -- the placement engine and cost model -- is implemented and
tested standalone. What remains open is joining the two: `cim-placement`
has the algorithm and now has IR to run it on, but does not yet rewrite
that IR from its schedule.

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

## M3 — The placement pass (algorithm done, IR rewriting outstanding)
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
- [ ] `cim-placement` rewriting actual IR: the pass calls the engine, but
  recovering the use sequence from `cim.program`/`cim.mvm` and rewriting
  from the schedule is still `TODO`. Blocked on M2 — there is no IR to
  place until `cim-partition` emits some.
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
