# Roadmap

Milestones from the v0.1 spec (Section 13). Each has a public artifact —
nothing counts until it is public.

**Current state.** M0, M1, M2 and M3 are done. A `.onnx` file now compiles and runs:
`python/cim_frontend` reads an INT8 `MatMulInteger` model and emits the MLIR the
pipeline consumes, and the result is checked for exact int32 equality against
ONNX's own reference implementation (`docs/roadmap.md` M2's last box, closed).
Before that, every numerical claim this project made started from MLIR this
repository had written itself. All eight lowering passes are real,
though the last one covers a deliberately scoped slice (see below):
`cim-detect` annotates eligible matmuls, `cim-partition` lowers them
into per-tile `cim.program`/`cim.mvm` with partial-sum reduction and explicit
space transfers, `cim-placement` rewrites that IR from a Belady schedule --
eliminating redundant weight programming, assigning tile ids from the
solution rather than round-robin, and hoisting loop-invariant `cim.program`
ops out of an `scf.for` when doing so is provably safe -- `cim-schedule`
(Pass 4) inserts `cim.barrier` conservatively over that placed IR (see the M2
entry below for the placement rule and why it needs to see inside a loop
body, not just an `scf.for`'s own operand list), `cim-insert-transfers`
(Pass 5) inserts `cim.copy` wherever a `cim.mvm`'s activation is not already
`#cim.space<near>` and hoists that copy above an enclosing `scf.for` when the
source is loop-invariant (see the Pass 5 entry below -- on today's real
pipeline `cim-partition` already stages every activation into near space
itself, so this pass has nothing to do there and is tested on hand-written
IR instead), `cim-legalize-precision`
(Pass 6) inserts `cim.requantize` after every terminal accumulator with
`scale=1.0`/`zero_point=0` (no calibration step exists yet to derive anything
else from) and warns when the target's `output_effective_bits` clamps below 8
(see the Pass 6 entry below for the interpreter-side arithmetic and the known
`cim-partition` integration gap), `cim-lower-to-target`
(Pass 7) converts straight-line, single-tile cim ops into real `func.call`s
against `cimrt.h`'s C ABI so the result can go through MLIR's standard
`--convert-to-llvm` pipeline and come out as a real, linkable binary -- the
first pass in this project that has ever needed to CALL cimrt from generated
code rather than execute it from C++ (see the Pass 7 entry below for the
memory model this needed and the real-binary verification), and
`cim-cost-report`
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
once, however many inferences run. It does not exactly reproduce
`cim-bench`'s `mm-spill-*` numbers, and the size of that difference is now
measured rather than left as a disclaimer: on the `mm-spill-2x` shape (16
weight blocks over 8 tiles) this pass emits `7 + 9*T` programs -- 9007 at
1000 inferences, against the simulator's optimal 8538. **5.5%**, not the
16000 the old phrasing here ("does not reproduce the spill numbers") could
easily be read as. `cim-cost-report` now prints both numbers and the gap
between them, so this stays measured instead of asserted.

The residual 5.5% is not a missing optimization. `7 + 9*T` is `tiles-1`
hoisted plus `blocks-(tiles-1)` per iteration, which is the proven optimum
for any single loop body -- a weight no op in the body programs is never
written and therefore holds a tile permanently, at most `tiles-1` weights
can do that, and everything else is reprogrammed every iteration. The
flattened solve beats it only by varying its per-iteration program count
(8 on some iterations, 9 on others), and a fixed loop body cannot emit a
varying number of ops. See the M3 checkbox below for the measurement and
`the_n_inference_optimum_is_not_a_constant_per_iteration_count` in
`test/unit/placement_test.cpp` for the test that pins it.

That per-body optimum used to fall out of a tie-break in Belady's victim
scan rather than being aimed at -- `test/Transforms/cim-placement-spill-loop.mlir`
exists to catch the silent regression to `16*T` that a reasonable-looking
cleanup of that scan would otherwise cause. It is now reached
deliberately: `cim::computeSteadyStatePlacement`/`validateSteadyStatePlacement`
(`lib/Placement/Placement.h`/`.cpp`) compute a pin-and-stream schedule --
pin the `tiles-1` most-used weights to permanent tiles by occurrence count,
stream everything else through the one remaining tile -- and
`placeBlock` (`lib/Transforms/CIMPlacement.cpp`, decision 5) uses it
whenever its cost is strictly cheaper than the ordinary per-block solve's
own hoist. Proved exactly optimal, not just no-worse, when every weight in
a body is used once (`cim-partition`'s own shape, matching every workload
in `bench/workloads/`) via exhaustive search over every permutation up to
7 distinct weights against every tile count
(`steady_state_matches_brute_force_when_each_weight_is_used_once` in
`test/unit/steady_state_property_test.cpp`); when a weight repeats within
one body the pin choice is a heuristic (the true optimum there is a
residency fixed-point problem this project does not solve exactly), but
proved never worse than pinning nothing and always a genuine replay fixed
point, on both exhaustive small alphabets and 500 random instances.
`test/Transforms/cim-placement-deliberate-hoist.mlir` proves the real pass
reaches for it on the exact case where the old accidental tie-break failed
(`[A, B, C, A, A, D]`, A repeated three times, over 2 tiles), checked both
structurally (FileCheck) and numerically
(`placement_never_changes_values_with_a_weight_repeated_in_one_body` in
`test/mlir/pipeline_e2e_test.cpp`).

**Composition hardening.** Individually-real passes are not the same claim
as a pipeline that composes, and for a while this project made the former
claim while only the latter was true up to Pass 5. Three real composition
breaks got fixed (see the Pass 6 and Pass 7 entries above for the actual
fixes): `cim-legalize-precision` retyping a downstream `cim.copy`/
`memref.copy` chain where it safely can and falling back to a
width-preserving requantize where it can't (rather than tripping the
dialect verifier); `cim-lower-to-target` folding an identity
`memref.subview` of a device-space value through to its source handle
(rather than refusing the exact shape `cim-partition` emits for a
single-K-tile activation); and `cim-lower-to-target` lowering
`cim.requantize` itself against a new `cimrt_requantize` ABI call, so
`cim-legalize-precision` (which always inserts a real `cim.requantize`)
and `cim-lower-to-target` (which used to refuse it outright) can finally
sit in the same chain. `test/Transforms/cim-pipeline-full.mlir` proves all
three RUN lines: the two narrower chains kept as their own regression
coverage (through `cim-legalize-precision` and `cim-cost-report`, and
through `cim-cost-report` and `cim-lower-to-target` skipping precision
legalization), and a third (`FULL`) chaining literally all eight passes,
spec order, on one module, down to real `cimrt_mvm`/`cimrt_requantize`
calls with no cim ops left. `test/mlir/pipeline_e2e_test.cpp`'s
`e2e_full_pipeline_through_precision_legalization_composes` checks the
same composed chain's requantize clamp numerically against an
independently computed reference, through the interpreter (which stops at
the point `cim-lower-to-target` would take over); `test/real-target/`'s
`requantize-correct`/`requantize-wrong` pair is the same claim checked one
level further down -- a real compiled binary's `cimrt_requantize` call
clamping a genuine out-of-range value (40000, an 8-bit signed clamp can
only hold `[-128, 127]`) and genuinely trapping when the expected value is
wrong.

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

## M2 — Functional correctness (complete)
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
- [x] Pass 4 `cim-schedule`: v0.1 keeps source order and inserts
  `cim.barrier` conservatively rather than reordering or overlapping
  anything (double-buffering `cim.program`/`cim.mvm` is v0.2, gated on the
  target's `capabilities.double_buffer_program`, `docs/target-format.md`).
  Tracks one open, un-barriered run of results per `!cim.device`, and
  flushes it (a `cim.barrier` immediately before the dependent op) the
  moment anything actually reads one of those results -- as a direct
  operand, or from inside a region a candidate op owns. That second case is
  load-bearing, not defensive: an `scf.for`'s own operand list is just its
  bounds, never a value only its body references, so a hoisted
  `cim.program`'s resident being read by a `cim.mvm` inside the loop is
  invisible to a check of the `scf.for` op's direct operands alone, and
  missing it would put the barrier after the loop instead of before it --
  exactly the bug a first pass at this algorithm had, caught by
  `test/Transforms/cim-schedule.mlir` before it shipped. Verified two ways
  that do not overlap: the FileCheck suite pins exact barrier placement
  (including the "barrier before the loop AND once per iteration inside
  it" case above, and that unrelated bookkeeping -- a fresh `memref.alloc`,
  a `cim.tile_alloc` on a different device -- never forces an unnecessary
  flush), and `test/mlir/schedule_e2e_test.cpp` confirms scheduling never
  changes a computed value. Numerical invariance is necessary but not
  sufficient here: `cim.barrier` is a no-op in the functional simulator, so
  a broken scheduler and a correct one compute identical numbers -- a
  mutation test (reverting the nested-region check) confirms the FileCheck
  suite fails while the numerical suite still silently passes, which is
  exactly why both exist.
- [x] Pass 6 `cim-legalize-precision`: inserts `cim.requantize` after every
  "terminal" i32 accumulator -- a `cim.reduce_partial` result, or a
  `cim.mvm` result no `cim.reduce_partial` consumes (the single-K-tile
  case, nothing to reduce). `scale=1.0`/`zero_point=0` always: v0.1 has no
  per-layer calibration step anywhere in the pipeline to derive anything
  else from, and inventing calibration data this pass has no way to
  validate would be exactly the "silently produce a wrong-but-plausible
  number" this project's passes refuse to do elsewhere. What lands here is
  the op shape spec Sec. 6 calls for, and the clamp that
  `output_effective_bits` genuinely does encode -- on a target declaring
  fewer than 8 effective bits, the pass emits a warning naming exactly how
  many of the 256 i8 encodings are unreachable through that readout path
  (modeling analog ADC resolution loss, spec Sec. 5.3), rather than
  silently degrading. Idempotent: a terminal value already read by a
  `cim.requantize` is left alone rather than stacking a second one.

  `lib/Interpreter/Interpreter.cpp` gained real execution of
  `cim.requantize` -- round-half-away-from-zero (`quantized = zero_point +
  round(value / scale)`), then clamp to the signed range `effective_bits`
  can hold -- replacing what had been a deliberate "not implemented yet"
  stub error (guessing a rounding mode before this pass pinned one down
  would have produced numbers that looked right). Unlike `cim.barrier`,
  `cim.requantize` genuinely changes values, so there is no "does not
  change the answer" invariant to fall back on for verification: instead
  `test/mlir/legalize_precision_e2e_test.cpp` checks the interpreter's
  output against an independently-written reference implementation of the
  same formula, over a fractional scale and nonzero zero_point, negative
  rounding ties, and the `effective_bits` clamp hit from both directions.
  `test/Transforms/cim-legalize-precision.mlir` covers pass-level structure
  (which accumulators get requantized, which don't, idempotency, and the
  sub-8-bit warning) via FileCheck, on hand-written IR rather than the live
  pipeline -- see the integration-gap note below.

  Composes with `cim-partition`'s output. `cim-partition` always emits its
  own `cim.copy` back to a host buffer typed to match the *original* i32
  accumulator, with no knowledge that a later pass might requantize that
  value down to i8 -- naively rewiring that copy's operand to a narrower
  result would leave its declared result type stale and trip the dialect
  verifier ("copy must not change element type"). Rather than teach
  `cim-partition` to anticipate a requantize that may never run (coupling
  two otherwise-independent passes and misreporting buffer sizes whenever
  this pass is absent), `cim-legalize-precision` looks at what is actually
  downstream of each terminal before deciding how to narrow it: if every
  consumer is something the pass is free to retype in place (another
  `cim.copy`, or a `memref.copy` into a `memref.alloc` it owns, or a
  `memref.dealloc`, which has no separate declared type to go stale), it
  narrows all the way to i8 and retypes that whole local chain to match.
  Otherwise -- the case on real `cim-partition` output, where the
  copy-back's ultimate destination is a subview of the function's own
  output argument, a type this pass has no license to change -- it falls
  back to a requantize whose result type matches the terminal's *original*
  element type exactly. That is not a lesser form of legalization:
  `cim.requantize`'s clamp to `effective_bits` is a value-range operation
  (`Interpreter.cpp`'s `runRequantize` already supports any integer result
  width), so the readout-path accuracy loss spec Sec. 5.3 cares about is
  modeled correctly either way -- only the storage container's width
  differs, and only when narrowing it would require changing a type this
  pass does not own. Both shapes are covered by
  `test/Transforms/cim-legalize-precision.mlir`
  (`narrows_through_a_locally_owned_copy_chain` and
  `falls_back_when_the_sink_is_not_owned`), and the composed case against
  real `cim-detect`/`cim-partition` output is exercised end to end by
  `test/Transforms/cim-pipeline-full.mlir`.
- [x] Pass 5 `cim-insert-transfers`: `cim.mvm`'s activations operand must
  live in `#cim.space<near>` (spec Sec. 3.4) -- not yet enforced by
  `MvmOp::verify()` itself, precisely because this is the pass responsible
  for making it true. Wherever an activation is not already near-space,
  inserts a `cim.copy` and rewires the `mvm` to read the copy's result.
  Followed by the "small dataflow analysis to hoist redundant copies out of
  loops" spec Sec. 6 also calls for: when the activation being copied is
  loop-invariant with respect to the `mvm`'s nearest enclosing `scf.for`
  (its home region is not the loop body or anything nested inside it), the
  copy is inserted once, immediately before the loop, rather than once per
  `mvm` per iteration -- and shared by every `mvm` inside that loop reading
  the exact same invariant source, rather than duplicated per site.
  Idempotent: an activation already in near space is left alone, which is
  also what makes re-running the pass a no-op.

  `cim-partition`'s own output already stages every activation into near
  space itself (see its file header) before this pass would ever see it,
  so on today's real pipeline there is nothing for it to do -- unlike Pass
  6's note above, this is not a composition gap to close: there is
  genuinely no near-space-staging work left for this pass on real
  `cim-partition` output, by design. Tested on hand-written IR that
  manufactures the space mismatch cim-partition's current output never
  produces (`test/Transforms/cim-insert-transfers.mlir`): a host-space
  activation gets a copy, an already-near one is left alone, a
  loop-invariant activation is hoisted above the loop exactly once and
  reused by every `mvm` in that loop sharing it, and a loop-variant
  activation (computed from the induction variable via a `memref.subview`)
  stays inside the loop, right before its `mvm`. `cim.copy` is a faithful
  byte-for-byte move, not a value-changing operation, so numerical
  verification is an invariance check like `cim-schedule`'s rather than an
  arithmetic differential: `test/mlir/insert_transfers_e2e_test.cpp`
  confirms running a module with and without the pass computes identical
  numbers. That invariance is necessary but not sufficient on its own,
  same as `cim-schedule`'s barrier placement: a mutation reverting the
  loop-invariance check produces correct numbers with a redundant copy
  re-inserted inside the loop on every iteration, which the numerical
  suite alone would silently accept -- only the structural FileCheck
  suite catches it.
- [x] Pass 7 `cim-lower-to-target`: converts cim ops into real `func.call`s
  against cimrt.h's actual C ABI, so the result can go through MLIR's
  standard `--convert-to-llvm` pipeline, `mlir-translate`, and a linker,
  and come out as a real binary. Nothing else in this project had needed to
  CALL cimrt from generated code before this -- the interpreter executes
  cim ops by calling cimrt directly from C++, one op at a time, as it
  walks the IR; this pass is that same job done at compile time, reusing
  the interpreter's own staging model (fresh scratch buffer per call,
  freed once its one use is done) for the same reason.

  v0.1 scope is deliberately narrow, agreed with before starting rather
  than discovered partway through: straight-line code only -- any cim op
  nested inside a region (an `scf.for` body, exactly what
  `cim-placement`'s own loop hoisting produces) was refused with a
  diagnostic, since a buffer this pass allocated would either leak every
  iteration or need a hoisting analysis of its own, neither designed
  here. **That is no longer true for plain `scf.for` nesting, at any
  depth** -- see the loop-body lowering paragraph below, added in the
  same session as the multi-K-tile closure and later generalized past its
  original one-level limit (M4 above). `cim.reduce_partial`, originally
  refused
  alongside it for a stated reason ("multi-tile K-reduction needs its own
  buffer-lifetime story this pass does not have -- multiple partial-sum
  buffers alive at once, reduced into one"), turned out to overstate the
  problem the same way `cim.requantize`'s original refusal did (below):
  the N partial-sum buffers are not fresh scratch this pass would need to
  juggle, they are already-live `cim.mvm` results this same pass lowered
  earlier in the straight-line walk, exactly like any other device-space
  value that outlives its producing op. It is lowered now
  (`lowerReducePartial`, `lib/Transforms/CIMLowerToTarget.cpp`) against a
  new `cimrt_reduce_add` ABI call (`runtime/include/cimrt.h`) that sums
  two device buffers elementwise with wrapping (not saturating)
  addition -- matching `Interpreter.cpp`'s `runReducePartial` bit for
  bit, the same "two independent implementations of one contract must
  agree exactly" discipline `cimrt_requantize`'s rounding mode already
  follows. An N-operand reduce becomes N-1 chained calls, with each
  INTERMEDIATE accumulator this function itself allocates freed once
  consumed; only the final one survives as the op's result -- that
  chaining is the one piece of real bookkeeping this lowering actually
  needed, considerably smaller than the original refusal's framing
  suggested.

  Composing this with a real multi-K-tile matmul from `cim-partition`
  needed one more fix, closed in the same session: `cim-partition` slices
  ONE shared staged activation buffer per K-tile (`memref.subview` with a
  nonzero offset for every tile past the first), and a non-identity slice
  of a device-space buffer used to have no lowering here at all. It does
  now: a new `cimrt_copy_range` ABI call (`runtime/include/cimrt.h`) --
  a byte-range generalization of `cimrt_copy` the same way `cimrt_write`/
  `cimrt_read`'s own offset parameters generalize a whole-buffer transfer
  -- materializes a rank-1, unit-stride slice of a rank-1 source into a
  fresh buffer of its own size. `checkAllowedConsumers` computes that byte
  range while the slice's SOURCE type is still real and records it in a
  new `materializedSliceRange` map for `lowerSubview` to use once it no
  longer can (mirroring how `deviceValueElemBits` already closes the
  analogous gap for element width); a genuinely higher-rank or
  non-unit-stride slice is still refused, since it has no
  `cimrt_copy_range` equivalent (one contiguous byte range) and is a real,
  separate ABI question. Closing this is what finally lets a REAL
  multi-K-tile matmul reach `cimrt_mvm` through this pass at all --
  `test/Transforms/cim-pipeline-multi-k-tile.mlir` runs cim-detect,
  cim-partition, and cim-lower-to-target (and, in its FULL variant,
  literally all eight passes) on a genuine `linalg.matmul_transpose_b`
  spanning two K-tiles and checks it reaches `cimrt_copy_range`/
  `cimrt_mvm`/`cimrt_reduce_add` with no `cim` ops left, and
  `test/real-target/multi-k-tile-correct`/`-wrong` takes that same real
  `cim-detect`/`cim-partition`/`cim-lower-to-target` chain all the way
  through `mlir-translate`, `clang`, and a linker into an actually-run
  binary (an all-ones 4x8 weight against activation `[1..8]`: every output
  row must sum all eight activation values to 36, which only happens if
  both K-tiles' contributions -- and therefore both the slice and the
  reduce -- are genuinely combined; 10 or 26 would mean one K-tile's
  contribution was silently dropped). `cim.requantize`, originally
  refused for the same stated reason as `cim.reduce_partial`, is
  also now lowered (see the composition-hardening entry above): it turned
  out to have no such buffer-lifetime problem in the straight-line,
  single-tile slice, since `cim-legalize-precision` makes it a terminal
  accumulator's SOLE consumer -- one producer, one consumer, exactly like
  `cim.mvm`'s own staged activation.

  Memory model: a `#cim.space<near|insitu>` memref is not a real memref
  after this pass runs -- cimrt's buffers are opaque handles, incompatible
  with the raw-pointer descriptor standard memref-to-llvm lowering
  assumes -- so every such SSA value is replaced end-to-end by the
  `!llvm.ptr` (a `cimrt_buffer*`) `cimrt_alloc` returned for it, while
  `#cim.space<host>` memrefs stay real, with bytes reaching cimrt only via
  `cimrt_write`/`cimrt_read` and an extracted raw pointer. Rewriting
  happens strictly in program order with each op erased as it is lowered,
  so a downstream op's operand is already the new value by the time this
  pass inspects it -- which is also what makes an operand's current type
  self-describing (still a memref means real host memory; already
  `!llvm.ptr` means an already-staged device buffer, the common case on
  real pipeline output since `cim-partition` already stages activations
  into near space itself) with no separate bookkeeping needed, except for
  one thing an operand's type genuinely cannot recover: which device a
  bare i32 tile id belongs to. A `memref.dealloc` on a device-space value
  becomes `cimrt_free`; a still-live device-space value with no such
  dealloc anywhere in the IR leaks, exactly as it would in hand-written C
  using cimrt directly with no free -- a known, documented limitation (no
  liveness analysis here), not an oversight.

  Composes with `cim-partition`'s output. Slicing a staged activation per
  K-tile (the single-tile v0.1 case: `cim-partition` still emits a
  `memref.subview` even when there is only one tile) initially broke
  composition independently of Pass 6's own gap: `checkAllowedConsumers`
  only recognized `cim.program`/`cim.mvm`/`cim.copy`/`memref.dealloc` as
  consumers of a device-space value, so the subview tripped its refusal.
  Fixed by recognizing an IDENTITY subview specifically (offset 0, full
  extent, unit stride, no rank reduction -- verified against the source's
  actual static shape, never assumed) and folding it straight through to
  the same handle once the pass reaches it in program order, rather than
  allocating a second buffer or attempting a real device-side slice.
  Reached only via `checkAllowedConsumers`, which is what lets `lowerSubview`
  fold unconditionally once a subview's operand is already `!llvm.ptr`: any
  non-identity subview -- the real multi-K-tile case -- is refused there,
  with its own diagnostic, before `lowerSubview` is ever reached, rather
  than silently reading the wrong bytes. `cimrt_mvm` has no offset or
  sub-buffer concept in its ABI, so a genuine slice of a device-space
  buffer needs a real ABI decision (a sub-buffer notion, or staging each
  slice separately) that is real, separate work -- tracked as M4, not
  attempted here. `test/Transforms/cim-lower-to-target.mlir`'s
  `identity_subview_of_a_device_value_folds` and
  `non_identity_subview_is_refused` cover both directions, and
  `test/Transforms/cim-pipeline-full.mlir` exercises the fold against real
  `cim-partition` output. The non-identity case's own refusal did not
  stay refused: it is lowered now too, against a new `cimrt_copy_range`
  ABI call -- see the `cim.reduce_partial` entry above, which is where
  that fix actually landed (both were needed together to let a real
  multi-K-tile matmul reach this pass at all). `non_identity_subview_is_refused`
  itself was renamed `non_identity_rank1_slice_is_materialized` to match;
  a new `non_identity_higher_rank_slice_is_still_refused` covers what is
  still refused.

  `cim.requantize` lowering (originally refused outright, see the v0.1
  scope paragraph above). Against a new `cimrt_requantize` ABI call
  (`runtime/include/cimrt.h`, `runtime/src/simulator/simulator.cpp`) that
  does the round-half-away-from-zero and signed-clamp arithmetic
  device-side, rather than reproducing that arithmetic a second time in
  generated IR -- this pass has never computed a value itself anywhere
  else, it stages memory and dispatches calls, and a second independent
  copy of that formula would risk silently drifting from
  `Interpreter.cpp`'s own. `lowerRequantize` mirrors `lowerMvm`'s shape
  closely (`stageForRead` for the input, `allocBuffer` for the output,
  call, `checkOk`, branch on the result's declared space). The one new
  piece of bookkeeping: `cimrt_requantize` needs the input's element width
  independently of the (always-safe, own) result type, and an
  already-lowered device-space input operand is a bare `!llvm.ptr` with no
  width of its own -- `deviceValueElemBits` (a `Value -> unsigned` map,
  populated everywhere `lowerMvm`/`lowerCopy`/`lowerRequantize` hand back a
  device-space handle) recovers it, the same kind of gap `tileDevices`
  already existed to close for tile ids. Verified three ways:
  `test/Transforms/cim-lower-to-target.mlir`'s
  `requantize_host_narrows`/`requantize_device_narrows_and_stays_a_handle`
  structurally; `test/Transforms/cim-pipeline-full.mlir`'s `FULL` RUN line
  against real, fully-composed `cim-partition` output (see the
  composition-hardening entry above); and `test/real-target/`'s
  `requantize-correct`/`requantize-wrong` pair, a real compiled binary
  whose `cimrt_requantize` call clamps a genuine out-of-range value
  (40000, an 8-bit signed clamp can only hold `[-128, 127]`) and genuinely
  traps when the expected result is wrong -- the strongest evidence this
  project has that the composed chain is a working artifact, not just
  well-shaped IR. `cimrt_requantize` itself is checked directly in
  `test/unit/cimrt_test.cpp` against an independently-written reference,
  covering narrowing, widening, rounding ties, and the clamp firing in
  both directions; **not yet counted by `cimrt_profile_stop`** -- the
  target schema's `costs:` section has no requantize/readout entry, a
  known simplification (spec M4), not a silent omission.

  **Loop-body lowering** (originally one level of `scf.for` nesting,
  closed in a later session than the multi-K-tile work above; generalized
  to arbitrarily many levels of plain `scf.for` nesting in a later
  session still -- see the M4 entry above for that part): `run()` now
  recognizes a plain `scf.for` (no `iter_args`) sitting directly in a
  function's entry block, or nested inside another such recognized loop
  at any depth, as a lowering target alongside the entry block itself --
  exactly the shape `cim-placement`'s own loop hoisting produces once a
  matmul's activation has more than one row (recall `cim-partition`
  implements the matrix-*vector* contract only; a multi-row loop is
  always something the CALLER wrote, never something `cim-partition`
  introduces itself -- `test/Transforms/cim-placement-loop.mlir`'s
  `@fits_hoists_entirely` is exactly that hand-written shape).
  `collectRecognizedLoopBodies` recurses through the whole nest to find
  every such loop's body; an `scf.if`, or a loop with loop-carried
  `iter_args`, containing a cim op is still refused with a diagnostic at
  any depth, matching the discipline every other scope limit here
  follows.
  
  The actual hoisting: every buffer this pass allocates while lowering a
  recognized loop nest's body -- an activation-stage `cim.copy`'s device
  buffer, `cim.mvm`'s result, a readback `cim.copy`'s host buffer, and so
  on for every other op's own buffer creation site -- is created ONCE,
  immediately before the OUTERMOST loop in the nest (`creationBuilder`,
  `FunctionLowering`; `loopHoistBefore` is set once on entering that
  outermost loop and left unchanged while recursing into anything nested
  inside it), instead of at the un-lowered op's own position inside it.
  Every loop body in the nest keeps only the write/compute/read calls
  that use the result, an ordinary SSA value any level can reference with
  no special capture (`scf.for`'s region has no capture-list concept to
  begin with, and a value defined before the outermost loop dominates
  every level nested inside it). This is what keeps a real, multi-row (or
  multi-row-and-column) matmul from leaking one buffer's worth of device
  memory per innermost iteration: the SAME handle is reused (overwritten)
  on every iteration of every nesting level, matching how actual
  scratchpad hardware would behave, rather than a fresh one being handed
  out and abandoned every time. A buffer's matching free -- either this
  pass' own `cimrt_free` for a device handle, or a real `memref.dealloc`
  for a host one -- is, symmetrically, deferred to just after the
  OUTERMOST loop if and only if that specific buffer's own allocation was
  itself hoisted (`hoistedThisLoop`, a `Value` set scoped to the current
  loop nest's lowering, not just whichever level is being visited):
  freeing the one shared, reused buffer on whichever iteration the
  original free happened to sit on would make every later iteration's use
  of it a use-after-free. A buffer this pass both allocates AND frees
  within one op's own lowering (e.g. `stageForRead`'s scratch buffer,
  `cim.program`'s weight-staging buffer, a non-final `cim.reduce_partial`
  accumulator) is hoisted exactly the same way, for the same reason --
  reused every iteration instead of realloc'd -- and its free just moves
  with it, to the same after-outermost-loop position, rather than needing
  separate treatment.
  
  Verified four ways: `test/Transforms/cim-lower-to-target.mlir`'s
  `loop_hoists_and_reuses_its_buffers` covers the one-level positive case
  structurally (every `cimrt_alloc` and the host readback's
  `memref.alloc` appear once, before `scf.for`; the write/mvm/read calls
  reuse the same handles inside it), and
  `doubly_nested_loop_hoists_to_the_outermost_loop` covers the same claim
  one level deeper -- the buffers this time must appear once before the
  OUTER `scf.for`, not the inner one, and the deferred frees must land
  after BOTH closing braces; `loop_defers_a_hoisted_device_buffers_free`
  and the first test's own host-buffer check cover the deferred-free
  mechanism directly (an explicit `memref.dealloc`/free inside the loop
  body, on a buffer this pass hoisted, ends up relocated to just after
  the loop -- not exercised by real pipeline output today, since neither
  `cim-partition` nor `cim-placement` currently emit such a dealloc, but
  worth guaranteeing correct rather than leaving it an untested, silently
  wrong case waiting for the day something does); `loop_with_carried_values_is_refused`,
  `cim_op_inside_an_scf_if_is_refused`, and
  `cim_op_inside_an_scf_if_is_refused_at_any_loop_depth` cover what is
  still refused. Beyond the structural suite, a genuine multi-row `linalg.matmul_transpose_b`
  (identity weight, three rows with distinct values) was run through the
  real `cim-detect`/`cim-partition`/`cim-placement`/`cim-lower-to-target`
  chain, then all the way through MLIR's `--convert-to-llvm` pipeline
  (needing `--lower-affine` and `--convert-scf-to-cf` for the first time
  here -- a dynamic per-iteration `memref.subview` offset becomes an
  `affine.apply` nothing else converts, and `cf.assert`'s own multi-block
  lowering must not run before `scf.for` itself has become real control
  flow, or `scf.for`'s single-block verifier rejects the result),
  `mlir-translate`, `clang`, and linked against the real
  `runtime/libcimrt.a`, then actually **run** as a native binary
  (`test/real-target/check-loop.mlir.in`, `loop-correct`/`loop-wrong`): a
  checksum combining all three rows' first output element (951) only
  comes out right if all three loop iterations independently computed
  and kept their own correct value rather than reusing or clobbering a
  stale one -- the strongest evidence yet that the buffer reuse this
  paragraph describes is a working artifact on real, compiled, executed
  hardware-shaped code, not just well-typed IR. The doubly-nested case has
  its own real-binary proof too, hand-written rather than produced by
  `cim-placement` (which never nests two levels itself, M3 above):
  `test/real-target/check-nested-loop.mlir.in`
  (`nested-loop-correct`/`nested-loop-wrong`) hand-writes `cim.device_open`/
  `cim.program`/`cim.mvm`/`cim.copy` directly inside a doubly-nested,
  plain `scf.for`, six (2 outer x 3 inner) iterations each with a distinct
  first activation element, checked against a checksum (654321) that only
  comes out right if all six iterations, across both loop levels,
  independently computed and preserved their own correct value.

  A real bug this design surfaced during its own gate run, worth keeping
  in mind for any future pass that touches already-rewritten operands: the
  ODS-generated typed accessors for a dialect-fixed operand type (e.g.
  `TileAllocOp::getDevice()`, `MvmOp::getWeights()`, and -- less
  obviously -- *any* `AnyMemRef`-constrained accessor too, such as
  `CopyOp::getSource()`) perform an unconditional cast to that declared
  type. Once this pass has replaced the actual value behind such an
  operand with a `!llvm.ptr`, calling the typed accessor is undefined
  behavior: silently fine in a release build (assertions compiled out),
  an immediate crash in a debug build (assertions on). It passed every
  release-build test and still crashed the first time it ran under
  `build-cov`'s Debug configuration. Fixed by reading every such operand
  through `op->getOperand(N)` instead, wherever the operand might already
  have been rewritten by an earlier op this same pass lowered.

  Verified two ways: `test/Transforms/cim-lower-to-target.mlir`'s
  structural FileCheck suite covers every real op, every `cim.copy` space
  combination (host->device, device->host, device->device, host->host),
  the device-space-dealloc-to-`cimrt_free` translation, and every refusal
  diagnostic. Beyond that -- the strongest verification available for a
  pass whose entire point is producing something a compiler pipeline can
  run for real -- this pass's output for a case shaped like the
  straight-line FileCheck test was taken all the way through MLIR's
  standard `--convert-to-llvm` pipeline, `mlir-translate`, `clang`, and
  linked against the real `runtime/libcimrt.a`, then actually **run** as a
  native binary: it computed the correct `cim.mvm` result against real
  (simulated) hardware, not the interpreter, and a deliberately wrong
  expected value in that same check made the binary genuinely trap
  (`cf.assert` firing a real `SIGABRT`) rather than silently pass. First
  verified once by hand, that round trip is now reproducible on demand as
  `test/real-target/` (`-DCIM_ENABLE_REAL_TARGET_E2E=ON`, default OFF --
  it needs `mlir-opt`, `mlir-translate` and `clang` specifically, a linker
  and a target triple the main suite has no business depending on to
  configure at all): a `cim.mvm` against an identity weight, with a
  `cf.assert` baked directly into the source MLIR checking one element of
  the real result against a compile-time constant, built into two
  binaries that differ only in that constant. `ctest -R real-target` runs
  both -- the correct-value binary must exit 0, and the wrong-value one
  must genuinely crash (a `sh -c '! ...'` wrapper, since CTest's
  `WILL_FAIL` explicitly does not invert a signal-terminated failure) --
  so both directions of "the assertion works" stay checked, not just the
  happy path. Still not part of the main gate or CI (same reasoning as
  before: a linker and target triple this suite should not require to
  configure), but no longer only documented in a file header either.
- [x] End-to-end: an ONNX INT8 matmul compiles and produces numerically
  correct output, checked against ONNX's own reference implementation
  rather than PyTorch. `python/cim_frontend` reads a `.onnx` file and
  emits the MLIR the pipeline consumes; `test/python/test_onnx_frontend.py`
  runs the model through `cim-opt` and `cim-run` and requires exact int32
  equality with the oracle.

  **The oracle is `onnx.reference.ReferenceEvaluator`, not PyTorch**, and
  that is an improvement on this box's original wording rather than a
  shortcut. It ships inside the `onnx` package, and it is the
  specification's own reference implementation -- written by the people
  who write the spec, which is precisely the "written by other people for
  other reasons" property `test/python/conftest.py` names as the only one
  that makes a differential test worth running. PyTorch is ~800MB in CI
  and is a *producer* of ONNX rather than the spec's reference, so it
  would be both heavier and a weaker oracle. `onnxruntime` runs as a
  second, independent oracle where it is installed; the two must agree
  with each other as well as with us.

  What is verified: a graph of one `MatMulInteger` with a constant int8
  weight, swept over shapes that tile a target's geometry in N, in K, in
  both, and past the tile count. What is refused, loudly and with a test
  each: every other ONNX op, non-constant weights, `uint8` operands,
  non-zero zero points, more than one output row, symbolic shapes. The
  importer never emits a module it is unsure about -- silently dropping an
  unrecognized op would produce something that compiles, runs, and
  computes a different function than the model.

  One thing this deliberately does NOT do: run the full eight-pass chain.
  `cim-legalize-precision` inserts a `cim.requantize` clamping every
  accumulator to `precision.output_effective_bits` (8 in every shipped
  target), which is a real modeled hardware effect and means the result
  stops matching an unquantized oracle as soon as an accumulator leaves
  `[-128, 127]`. The differential runs detect/partition/placement, and
  says so at the point it builds the pipeline string.

  **Chained layers are also closed**: a graph of two or more `MatMulInteger`
  nodes, bridged by `Cast(to=float32) -> QuantizeLinear(scale=1.0,
  zero_point=0)` between consecutive layers, lowers to a real
  `cim.requantize(scale=1.0, zero_point=0, effective_bits=8)` sitting
  between the two compiled matmuls. Scale 1.0 is why this still matches an
  unquantized oracle exactly rather than only approximately: it makes
  ONNX's round-half-to-even and `cim.requantize`'s round-half-away-from-zero
  compute the same thing, because rounding an already-integer accumulator
  has no fractional part to round differently -- there is no tie for the
  two modes to disagree about. Verified by hand against a real
  `cim-opt`/`cim-run` round trip before the graph-walking code that finds
  this pattern was written. This is what makes the `mlp-3layer` benchmark
  shape reachable from a real model file, and it is the first time
  `cim-placement`'s reuse *across* layers (not just within one) is
  exercised from a model file rather than a generated one.

  Still not accepted: `Gemm`, `QLinearMatMul`, and quantize/dequantize
  graphs that are not this exact bridge (any other scale, or an absent
  zero point, reintroduces the rounding-mode risk the scale=1.0
  restriction exists to avoid).

### Known limits of cim-partition
Each of the remaining two is refused with a warning and the `linalg` op left
intact, so the module stays correct and is simply not offloaded:
- Only `linalg.matmul_transpose_b` (weights `[N x K]`, matching `cim.mvm`'s
  output-major convention). A plain `linalg.matmul` needs a transpose first.
- Only a single output row: `cim.mvm` is a matrix-vector primitive and the
  v0.1 contract is matrix-vector.

A third limit is closed: N and K no longer need to be exact multiples of the
tile geometry. Spec Sec. 6's zero-padding is implemented as the pad-and-copy
sequence this note used to say was missing -- when a dimension falls short
of the next tile multiple, a fresh `memref.alloc` host buffer of the padded
shape is zero-filled (`linalg.fill`) and the real weight/activation data is
copied into its top-left corner (`memref.copy`) before tiling proceeds
exactly as in the exact-multiple case. Padding rows/columns can only ever
contribute a `0 * x` term to the MVM they take part in, so they cannot
change any answer; the write-back is the one place raggedness still has to
be handled explicitly, cropping each block's (possibly padded)
tileRows-sized result down to however many of its rows are real before
copying it into `%out`. Verified both ways this project verifies a
composition: `test/Transforms/cim-partition.mlir`'s
`ragged_n_is_zero_padded`/`ragged_k_is_zero_padded` pin the exact IR shape
(the alloc, the fill, the copy, the crop), and
`test/mlir/pipeline_e2e_test.cpp`'s `e2e_ragged_n_is_zero_padded_and_cropped`/
`e2e_ragged_k_is_zero_padded`/`e2e_ragged_in_both_dimensions` execute the
padded IR through the interpreter and check it against the plain, unpadded
reference -- which has no idea padding happened, so any leak of a padding
row or column into a real output would show up as a wrong number, not just
a structurally-odd one. That numerical path needed one addition to the
interpreter itself: `linalg.fill` execution (`lib/Interpreter/Interpreter.cpp`),
since cim-partition's zero-padding is the only source of that op in this
pipeline and nothing previously modeled it. The scratch buffers this
allocates are never freed, matching this pipeline's existing convention:
none of cim-partition's other scratch buffers (the staged activation, each
block's host output copy) are freed either -- buffer lifetime management is
out of scope for v0.1 across the board, not just here.

## M3 — The placement pass (complete)
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
- [x] A full N-inference Belady solve on compiled IR. The solve itself now
  runs in `cim-placement` (`accumulateNInferenceOptimum`, decision 4 in
  that file's header): for a loop body with a compile-time-constant trip
  count it flattens the body's use sequence across all iterations, solves
  it with the same `cim::computePlacement` `cim-bench` uses, replays it
  through `validatePlacement`, and records the result on the module.
  `cim-cost-report` publishes it beside what the emitted IR actually
  costs, as `n-inference-optimum-programs` / `emitted-programs` /
  `placement-gap-percent`.

  The original wording of this box -- "matching what `cim-bench`'s
  simulator computes for the spill workloads" -- turned out to be asking
  for something impossible, and finding out why is the actual result here.
  Measured on the real spill shape (16 weight blocks over 8 tiles): the
  optimal schedule's per-iteration `cim.program` count is **not constant**.
  It emits 8 on some iterations and 9 on others (7 eights and 8 nines per
  15-iteration period, averaging the 8.538 the published 8538-per-1000
  figure comes from). A single loop body executes a fixed set of ops every
  iteration and therefore emits a constant number of programs per
  iteration, so no rewrite of one loop body -- including one that computed
  tile ids from the induction variable, which was the obvious candidate --
  can reach it. `the_n_inference_optimum_is_not_a_constant_per_iteration_count`
  in `test/unit/placement_test.cpp` pins that, because it is the argument
  against a much more invasive dialect change that would have bought
  exactly zero.

  What the gap actually is, which was never measured before: on the
  `mm-spill-2x` shape `cim-placement` emits `7 + 9*T` programs, so 9007 at
  1000 inferences against the simulator's 8538 -- **5.5%**, not the 16000
  a reader could reasonably infer from the old "does not reproduce the
  spill numbers" phrasing. `7 + 9*T` is `tiles-1` hoisted plus
  `blocks-(tiles-1)` per iteration, which is the proven optimum for any
  single loop body: a weight no op in the body programs is never written
  and so holds a tile permanently, at most `tiles-1` weights can do that,
  and the rest must be reprogrammed every iteration.
- [x] Make that per-body optimum **deliberate** rather than a consequence.
  Before this, `cim-placement` reached `tiles-1` hoisted only because
  Belady's victim scan starts at tile 0 and breaks immediately on a
  never-again next-use, so tile 0 absorbed the whole spill and tiles
  1..n-1 were each written exactly once -- precisely the
  `programCountPerTile == 1` condition that fills `hoistCandidates`. Any
  change that spread evictions across tiles dropped hoisting to zero and
  regressed this workload to `16*T`, i.e. exactly LRU --
  `test/Transforms/cim-placement-spill-loop.mlir` guards that regression
  and explains it at length, but a guard is not the same as aiming at the
  result on purpose. Now it is aimed at: `cim::computeSteadyStatePlacement`
  pins the `tiles-1` most-used weights by occurrence count (ties by
  first-use order) and streams everything else through the one remaining
  tile via a single-tile Belady sub-solve, and
  `cim::validateSteadyStatePlacement` proves the result replays as a
  genuine fixed point by re-running it through the existing
  `validatePlacement` three times. `placeBlock` computes both the ordinary
  per-block solve and this deliberate schedule for every constant-trip
  loop and takes whichever is strictly cheaper, so every existing test
  where the two already coincide is rewritten byte-for-byte unchanged, and
  the case where they didn't -- a weight repeated within one loop body,
  where the old tie-break's luck ran out -- now reaches the better
  schedule too:
  `test/Transforms/cim-placement-deliberate-hoist.mlir` (structural,
  `[A, B, C, A, A, D]` over 2 tiles, the exact case the old accidental
  mechanism hoisted nothing for) and
  `placement_never_changes_values_with_a_weight_repeated_in_one_body`
  (`test/mlir/pipeline_e2e_test.cpp`, the same shape checked against a
  numerical reference) are the proof on real IR;
  `test/unit/steady_state_property_test.cpp` is the proof on the engine
  itself, exhaustive for the provably-optimal once-per-body case and
  property-based for the heuristic repeated-weight case.
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
- [x] Volatile-vs-non-volatile comparison plot showing that persistence
  changes the optimum, not just the magnitude. Found on the way: the
  target schema's `standby_leakage_uw_per_tile` field
  (`docs/target-format.md`) was parsed and printed by `dump-target` but
  never actually fed into any cost number -- `amortizedInstallEnergyPjPerInference`
  divided `installEnergyPj` by `inferences` regardless of `persistent`, even
  though its own doc comment already said "on a volatile target the install
  cost recurs and this is a floor, not a limit." It is wired in now: a
  non-volatile target (`report.persistent`) amortizes exactly as before,
  since retaining its weights costs nothing once written; a volatile target
  additionally pays `numTiles * standbyLeakageUwPerTile` for as long as one
  steady-state inference takes (`report.steadyStateElapsedNsPerInference`,
  a new field -- the run's program-only latency plus its mvm latency, which
  nothing before this needed combined), and that term scales with elapsed
  time rather than with the install event, so it does not shrink as
  `inferences` grows. `test/unit/cost_report_test.cpp` checks the actual
  claim: the non-volatile curve keeps shrinking by ~10x per decade of
  `inferences` out to 10^12, the volatile one converges to (not just
  approaches) a computed floor and never drops below it, and a target
  declaring zero leakage reproduces the pre-fix formula exactly (backward
  compatibility, since `erbium-8t.yaml` itself declares
  `standby_leakage_uw_per_tile: 0.0`).

  `cim-bench` gained an `amortize` subcommand: one placement run (a single
  `CostReport`, since the amortization formula is closed-form over an
  arbitrary inference count -- re-simulating per sweep point would not
  change a single number and would not scale to the sweep's high end
  regardless) swept over a fixed log-spaced range of inference counts,
  emitted as JSON `points`. `bench/plots/plot_amortization.py` takes one
  non-volatile and one volatile run and plots both curves log-log: on
  `erbium-8t` vs. `generic-digital-cim` at the `mm-fit` workload, the two
  curves visibly cross -- the volatile target's cheaper install cost wins
  at low inference counts, the non-volatile target's amortization wins once
  its curve drops below the other's leakage floor -- which is the actual
  "changes the optimum" claim: a reader comparing the two targets at a
  single inference count would see only a magnitude difference and could
  not tell the crossover exists. `test/python/test_amortization_curve.py`
  is the second, independent check on the real shipped binary rather than
  the library directly: it asserts the non-volatile curve is still
  shrinking ~10x per decade at the end of the sweep, the volatile curve has
  visibly flattened (its last two points within 1%), and the two curves
  cross at least once.

## M4 — Second target and generalization (future)
- ~~`targets/generic-digital-cim.yaml` already exists as a placeholder
  second-class target file; needs real (or better-estimated) numbers and a
  working lowering to prove retargetability.~~ -- **closed, on the axis
  that actually mattered**: the `class:` enum itself turned out to already
  be incidentally covered (`test/targets/tiny-4x4.yaml` and
  `generic-digital-cim.yaml` are both `digital_cim`;
  `tiny-4x4-4bit.yaml` is `analog_cim`), and grepping every pass confirms
  none of them branch on `TargetClass` at all -- so the class tag was
  never actually a retargetability risk. The real, previously-untested
  axis was **persistence**: every compiler-level test target before this
  declared `persistent: true`, so `cim.program`'s own `persistent`
  attribute (set directly from `spec.tiles.persistent` in
  `lib/Transforms/CIMPartition.cpp`) had never been checked as `false`
  past the parser. `test/targets/tiny-digital-cim.yaml` mirrors
  `generic-digital-cim.yaml`'s real characteristics (volatile SRAM, no
  nonvolatile advantage, double-buffering capable) at tiny-4x4.yaml's
  scale, and now backs both a structural check
  (`test/Transforms/cim-partition-volatile.mlir`) and a full
  `real-target-e2e` binary pair
  (`digital-cim-volatile-correct`/`-wrong`) -- the strongest form of this
  proof, since it is a real compiled binary computing the right answer,
  not an attribute assertion. `targets/generic-digital-cim.yaml` itself
  still carries placeholder, estimated numbers (unchanged by this) --
  "better-estimated" real numbers for it remains open.
- `cim-legalize-precision` with real `effective_bits` modeling.
- ~~`cim-lower-to-target` beyond its v0.1 straight-line slice: lowering a
  cim op inside an `scf.for`~~ -- **closed**: one level of loop nesting
  (a single, plain `scf.for` with no `iter_args`, exactly what
  `cim-placement`'s own loop hoisting produces once a matmul's activation
  has more than one row) is lowered now, with every buffer this pass
  allocates while lowering the loop's body hoisted to before the loop and
  reused every iteration instead of leaking one buffer's worth of memory
  per iteration -- see the Pass 7 entry's own loop-body-lowering
  paragraph for the full design, and
  `test/real-target/loop-correct`/`-wrong` for a genuine multi-row matmul
  run as a real binary end to end. ~~A cim op nested any deeper than that
  one level (a second `scf.for`, an `scf.if`, or a loop with
  loop-carried `iter_args`) is still refused with a diagnostic~~ --
  **closed, for plain `scf.for` nesting**: `cim-lower-to-target` now
  recognizes and lowers a cim op nested inside a plain `scf.for` (no
  `iter_args`) nested inside another such loop, at any depth, not just
  one level -- `collectRecognizedLoopBodies`
  (`lib/Transforms/CIMLowerToTarget.cpp`) recurses through the whole nest
  instead of hardcoding a depth of one, and the hoisting design
  generalizes by hoisting every buffer this pass creates to just before
  the OUTERMOST loop in the nest (not merely the nearest enclosing one)
  and reusing it across every iteration of every level, freeing it once,
  after that same outermost loop. `cim-placement` itself still never
  *produces* a second level of nesting (v0.1 handles one, M3 above) --
  this closes what `cim-lower-to-target` can *lower* when handed such IR
  directly, the same way `cim.reduce_partial`'s and a genuine
  `memref.subview`'s lowering closed real gaps for shapes `cim-partition`
  already emitted before this pass could handle them. Structural proof:
  `test/Transforms/cim-lower-to-target.mlir`'s
  `@doubly_nested_loop_hoists_to_the_outermost_loop`. Real-binary proof:
  `test/real-target/nested-loop-correct`/`-wrong`, a genuine doubly-nested
  matmul (2 outer x 3 inner iterations) computed and checked end to end.
  Mutation-tested: forcing every recursion level to treat itself as
  outermost (hoisting to the nearest loop instead of the outermost one)
  was caught immediately by the structural test -- and, instructively,
  *not* by the real-binary test, which only proves numeric correctness,
  not hoisting depth; a buffer reallocated once per outer iteration
  instead of once total still computes the right answer, just less
  efficiently, so the structural test is the one that actually has teeth
  here.

  **Still open**: an `scf.if`, or a loop with loop-carried `iter_args`,
  containing a cim op is still refused with a diagnostic at any depth --
  that remains a real, separate design question (a loop-carried device
  value would need to survive as part of the loop's own iter_args type
  list, which this hoisting design says nothing about; conditional
  control flow has no notion here of which branch a hoisted buffer's
  single allocation should apply to).
- ~~`cimrt_requantize` accounted in the cost model: the target schema's
  `costs:` section needs a requantize/readout entry before
  `cimrt_profile_stop`/`cim-cost-report` can count it (currently zero-cost,
  a known simplification, not a silent omission -- see the Pass 7 entry
  above).~~ -- **closed, on the `cim-cost-report` side**: every shipped
  target file now declares `costs.requantize.latency_ns`/`energy_pj`
  (required, same as `program`/`mvm` -- an old file without it is rejected
  rather than silently charged zero), and `cim-cost-report`
  (`lib/Transforms/CIMCostReport.cpp`) walks and weights `cim.requantize`
  sites exactly like `cim.program`/`cim.mvm`, folding the result into
  `total_energy_pj`/`total_latency_ns`. This was a live gap, not a
  theoretical one: `cim-legalize-precision` inserts a `cim.requantize`
  after *every* terminal accumulator regardless of `output_effective_bits`,
  so any target run through the ordinary detect/partition/placement/
  schedule/transfers/legalize-precision/cost-report chain already had one
  silently uncounted (`test/Transforms/cim-pipeline-full.mlir`'s own
  PRECISION run is the existence proof, now with a `COST-JSON` check
  reading its cost-report output directly).

  **Closed, the other half of the same gap**: `cimrt_profile_stop`
  (`runtime/src/simulator/simulator.cpp`) now counts `cimrt_requantize`
  calls into a new `requantizes_issued` field of `cimrt_profile`
  (`runtime/include/cimrt.h`), charged against `costs.requantize` the same
  way `cimrt_program`/`cimrt_mvm` are charged against their own table
  entries. Getting there surfaced a real second gap, not just a missing
  counter: `Interpreter.cpp`'s `runRequantize` computed
  `cim.requantize`'s round/clamp arithmetic entirely in host code and
  never called `cimrt_requantize` at all, unlike `runProgram`/`runMvm`,
  which always go through `cimrt_program`/`cimrt_mvm`. That meant every
  interpreted `cim.requantize` executed for free -- adding the counter
  alone left `test/mlir/cost_report_e2e_test.cpp`'s new differential test
  failing (predicted 1, actual 0), which is what caught it. Fixed by
  making `runRequantize` delegate to `cimrt_requantize` exactly like the
  other two ops, which also collapses what used to be two independent
  copies of the same rounding arithmetic (host-side in the interpreter,
  device-side in the simulator) into one; a third, genuinely independent
  reference implementation in `test/mlir/legalize_precision_e2e_test.cpp`
  still checks it. New test:
  `cost_report_matches_runtime_on_a_requantized_module`; mutation-tested
  by disabling `CostAccumulator::recordRequantize`'s bookkeeping and
  confirming both it and `cimrt_test.cpp`'s
  `cimrt_profile_counts_a_known_trace` go red. `lib/Placement/CostReport.cpp`
  (the `cim-bench` engine, which models weight-programming amortization
  from a `PlacementResult` alone) is deliberately left out of this --
  see that file's own header and `docs/target-format.md`'s units note for
  why it has no notion of a requantize step at all.
- ~~`cim.reduce_partial` / `cimrt_reduce_add` is charged nowhere at all --
  no schema entry, no compile-time cost, no runtime cost.~~ -- **closed**,
  and with it the last of the three zero-cost-accounting holes: `cim.program`
  and `cim.mvm` were always charged, `cim.requantize` was closed by the two
  entries above, and this was the remainder. **Every op the runtime can
  execute is now charged against the target's cost table.**

  New required schema field `costs.reduce_partial.latency_ns`/`energy_pj`
  (`docs/target-format.md`, all 8 target files, the independent PyYAML
  oracle in `test/python/schema.py`, and `cim-bench dump-target`), required
  the same way `costs.requantize` is: an old file without it is rejected
  rather than silently charged zero.

  **The unit of charge is one `cimrt_reduce_add` CALL, not one
  `cim.reduce_partial` op** -- an N-operand reduce lowers to N-1 chained
  calls (`lowerReducePartial`), so `CostReportUtils.cpp` weights each site
  by `trip_count * (N-1)` and `CostAccumulator::recordReduceAdd` fires once
  per call. A reduce summing four partials therefore costs three times one
  summing two, which a naive per-site count would have flattened.

  As with requantize, the counter alone was not enough, and for the exact
  same reason: `Interpreter.cpp`'s `runReducePartial` recomputed the
  wrapping sum host-side in a `std::vector<int32_t>` loop and never called
  `cimrt_reduce_add`, so every interpreted reduce executed for free. Having
  been burned once, the differential test was written first and was red
  (predicted 1, actual 0) until `runReducePartial` was rewritten to stage
  its partials into device buffers and issue the same N-1 chained calls the
  compiled path does. That also collapses a second duplicated copy of the
  wrapping-add contract into one implementation. New test:
  `cost_report_matches_runtime_on_a_reduced_module` (a 4x8 weight over 4x4
  tiles -- two K-tiles, one two-operand reduce, exactly 1 call);
  mutation-tested by disabling `recordReduceAdd`'s bookkeeping and
  confirming both it and `cimrt_profile_counts_a_known_trace` go red.

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
