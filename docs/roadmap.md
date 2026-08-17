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
  exercised from a model file rather than a generated one. Scale 1.0 is
  no longer the only accepted value -- see M4 below for a real, calibrated
  scale.

  Still not accepted: `Gemm`, `QLinearMatMul`, and quantize/dequantize
  graphs that are not this exact bridge (an absent zero point, or a
  non-positive scale). A real, non-1.0 calibrated scale is now accepted
  -- see M4 below.

### Known limits of cim-partition
Everything genuinely out of v0.1's scope (a `linalg.conv_2d`-shaped op, a
batched matmul's 3rd dimension, dynamic shapes) is still refused with a
warning and the `linalg` op left intact, so the module stays correct and is
simply not offloaded -- this pass itself still has no convolution support,
and still should not. A *2-D convolution over a constant activation* is a
different claim, closed one layer up instead: see M4 below's ONNX
`QLinearConv` entry, which never emits a `linalg.conv_2d` at all -- it
reshapes the convolution into the SAME `linalg.matmul_transpose_b` this
pass already accepts, entirely in Python, before cim-detect ever runs. Both
weight-layout and row-count limits below are now closed:

~~Only `linalg.matmul_transpose_b` (weights `[N x K]`, matching `cim.mvm`'s
output-major convention). A plain `linalg.matmul` needs a transpose
first.~~ -- **closed, M4 below**: a plain `linalg.matmul` (weights in
`[K x N]`, ONNX's own layout) is now accepted too, transposed into a fresh
`[N x K]` buffer by this pass itself rather than refused.

~~Only a single output row: `cim.mvm` is a matrix-vector primitive and the
v0.1 contract is matrix-vector.~~ -- **closed, M4 below**: a matmul with
M > 1 rows ("batching") is now tiled the same way a single row is, wrapped
in a genuine `scf.for` this pass generates itself.

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
- **A gap this project's own audit found, not on the original list**:
  `capabilities.double_buffer_program` (and `partial_sum_in_place`,
  `autonomous_control`) were parsed and echoed by `cim-bench dump-target`
  but read by zero compiler passes -- the exact class of defect PR #3
  already found and fixed once for the `class:` enum. `cim-schedule`'s own
  header is honest about why: "nothing is reordered or overlapped... that
  is v0.2's double-buffering work" (spec Sec. 6 Pass 4) -- v0.1 never
  actually exploits the capability, so there was nothing for a pass to
  read.

  **Partially closed**: `lib/Placement/CostReport.cpp` (the `cim-bench`
  engine) now computes a PROJECTION -- `steadyStateElapsedNsPerInference
  IfOverlapped` (`CostReport.h`) -- of what the steady-state elapsed time
  per inference would be if a scheduler exploited
  `double_buffer_program`: `max(program latency, mvm latency)` for the
  window instead of their sum, the idealized assumption that whichever is
  smaller is fully hidden behind the larger. This is a comparison number,
  the same status `cim.n_inference_optimum`'s emitted-vs-optimal gap
  already has (M3 above) -- reported BESIDE the real, honest,
  never-overlapped figure in `formatCostReport`/`toJson`
  (`double_buffer_capable`, `steady_state_elapsed_ns_per_inference_if_
  overlapped`) and in `cim-bench amortize`'s JSON sweep
  (`amortized_install_pj_per_inference_if_overlapped`), never replacing
  it: the real number is still a strict, sequential reprogram-then-compute
  sum on every target, because that is genuinely what this compiler emits
  today regardless of what the hardware could do.

  Left at `0.0` (or, for the amortized figure, falls back to the honest
  serial value) on any target that does not declare the capability --
  same "leave it at zero rather than guess" discipline `requantizes`/
  `reduce_partial_adds` already follow in the same struct. Verified on the
  one shipped target that actually exercises it end to end:
  `generic-digital-cim.yaml` declares `double_buffer_program: true`, and
  `test/python/test_amortization_curve.py`'s new test runs the real
  `cim-bench` binary against it on the `mm-spill-8x` workload (mm-fit, the
  suite's default, reprograms nothing in steady state and would make the
  two curves trivially, correctly identical -- proving nothing about
  whether a real divergence renders end to end) and asserts the projected
  floor is strictly lower. `bench/plots/plot_amortization.py` draws it as
  a third, clearly-labeled dotted curve, only when the volatile input
  declares the capability. Unit-tested as a property (the projection is
  never greater than the serial sum, `cost_report_test.cpp`) and
  mutation-tested (swapping the `max` for a `+` was caught by both the
  property test and the amortization test, restored).

  **Still open**: this is the analytical engine only
  (`lib/Placement/CostReport.cpp`/`cim-bench`), deliberately not the
  runtime `CostAccumulator` (`runtime/src/simulator/cost_model.h`) or
  `cim-cost-report`'s IR-walking engine -- the functional simulator
  genuinely executes every `cimrt_*` call synchronously, one at a time
  (`cimrt_barrier` is a documented no-op: "Functional simulator executes
  synchronously; nothing to wait on"), so a real overlap number there
  would have to either misrepresent what actually ran or require an
  honest-to-goodness async execution model -- exactly the v0.2 scheduling
  work this capability is gated on, not a cost-model change.
  ~~`autonomous_control` remains fully unread -- v0.1's execution model
  has nothing host-less to drive at all.~~ -- documented as reserved, and
  now **pinned rather than merely claimed**: new fixture
  `test/targets/tiny-4x4-autonomous.yaml` is byte-identical to
  `tiny-4x4.yaml` except this one flag (true instead of false), and
  `test/Transforms/cim-autonomous-control-is-unread.mlir` runs the same
  real matmul through the full eight-pass-reachable chain
  (detect/partition/placement/schedule/insert-transfers/legalize-
  precision/lower-to-target) against both and diffs the two outputs byte
  for byte -- the strongest available claim that a flag is unread: not
  that one hand-picked pass ignores it, but that every pass's combined
  output is unaffected, character for character. Mutation-tested: an
  artificial one-character branch on the flag in
  `CIMLowerToTarget.cpp`'s `lowerDeviceOpen` was caught immediately by the
  diff, then reverted.

- ~~`partial_sum_in_place` remains fully unread.~~ -- **closed**: read by
  both `cim-lower-to-target` (`lowerReducePartial`) and the interpreter's
  `runReducePartial`. When a target declares it, an N-operand
  `cim.reduce_partial` allocates exactly ONE accumulator for the whole
  chain -- copied from the first partial via a new `cimrt_copy` call (the
  first partial's own buffer is never mutated directly: `stageForRead` can
  hand back an already-live handle that may have other uses this pass
  cannot see), then folded with the remaining N-1 partials via a new
  `cimrt_reduce_add_inplace` ABI call -- instead of one fresh buffer per
  chained step (`cimrt_reduce_add`). A new, separate ABI function rather
  than a relaxed `cimrt_reduce_add`, matching how `cimrt_copy_range` was
  added alongside `cimrt_copy` for a different new capability rather than
  loosening an existing, already-tested contract; charged identically to
  `cimrt_reduce_add` (same `costs.reduce_partial` entry, same
  `reduce_adds_issued` counter) since it is the same hardware step
  realized a different way. The interpreter learns the capability via a
  new `partial_sum_in_place` field on `cimrt_query`'s `cimrt_device_info`
  output (it never parses `TargetSpec` itself, unlike every compile-time
  pass, which already does independently) rather than a second,
  independent parse of the target YAML.

  Verified: `test/targets/tiny-4x4-no-inplace.yaml` (new -- every other
  tiny test target declares the capability true, so this is the one
  fixture that exercises "cannot accumulate in place" at all), structural
  FileCheck proof that a 3-operand reduce needs exactly one accumulator
  allocation with the capability versus two without it
  (`test/Transforms/cim-lower-to-target-reduce-partial-inplace.mlir`
  alongside the existing, now-explicitly-no-inplace
  `cim-lower-to-target.mlir`), direct `cimrt_reduce_add_inplace` unit
  tests mirroring `cimrt_reduce_add`'s own (hand-computed values, overflow
  wraps, invalid-argument rejection, identical cost accounting), and the
  existing `real-target-e2e-reduce-partial-correct`/`-wrong` binaries now
  genuinely exercise the in-place path end to end for free, since their
  shared target file already declared the capability.

  **A real, separate bug found along the way**: several existing
  FileCheck lines matched the bare substring `call @cimrt_reduce_add`,
  which is also a textual PREFIX of `call @cimrt_reduce_add_inplace` --
  `test/Transforms/cim-pipeline-multi-k-tile.mlir`'s checks were silently
  passing against the new function by accident once its target file
  (already `partial_sum_in_place: true`) started actually emitting it,
  rather than catching that the expected call had changed. Fixed by
  requiring the disambiguating `(` on every such check
  (`call @cimrt_reduce_add(`) and updating the expected call sequence to
  what the target's own declared capability now actually produces.

  **An honest limitation, found by mutation-testing**: forcing the
  interpreter to ignore `partial_sum_in_place` entirely (hardcoding
  either branch) breaks no test. Both branches compute the bit-identical
  wrapping-add result and charge the identical cost by design (the whole
  point of sharing `reduce_adds_issued`), so `cimrt_profile`'s counters
  cannot distinguish "took the correct branch" from "took the other,
  equally-valid one" from the outside -- unlike the COMPILED lowering's
  choice, which the structural FileCheck tests above do catch, since the
  two paths produce observably different IR. Mitigated, not solved: a
  dedicated test
  (`cost_report_matches_runtime_on_a_reduced_module_without_the_inplace_
  capability`) at least makes sure the general branch is reachable and
  correct through the interpreter at all -- before this session it had
  zero interpreter-level coverage, since every existing test target
  already declared the capability.
- ~~No packaging: no `pyproject.toml`/`setup.py`; the ONNX front end
  cannot be `pip install`ed despite being the project's front door~~ --
  **closed**. `python/pyproject.toml` packages `cim_frontend` with a real
  `cim-import-onnx` console-script entry point and an `onnx` extra
  mirroring `test/python/requirements-onnx.txt`'s split (numpy is a hard
  dependency; onnx/onnxruntime stay opt-in, matching why they were kept
  out of the C++ build entirely -- see `python/README.md`'s "Why Python,
  and why not in `tools/`"). `test/python`'s suites already import
  `cim_frontend` via a `sys.path.insert` hack that keeps working
  unchanged (source-tree imports still take priority), so packaging is
  additive: it does not touch how any existing test finds the module.

  Verified for real rather than just configured: a new `packaging` CI job
  (`.github/workflows/ci.yml`) does a fresh `pip install ./python[onnx]`
  in an environment with no MLIR toolchain and no source-tree
  `PYTHONPATH`, runs the installed `cim-import-onnx` console script
  (not `python3 -m cim_frontend`) against a model built with the
  project's own `test/python/onnx_fixtures.py` helper, and greps the
  emitted MLIR for the expected `linalg.matmul_transpose_b` shape --
  proving the packaged entry point does real importer work, not merely
  that `--help` exits zero. Confirmed locally in a scratch venv before
  writing the CI job.
- ~~The Section 17 units discrepancy is unresolved -- 1000x on program
  energy, documented in `target-format.md`, and it is the crux of the
  whole amortization argument~~ -- **not resolvable without hardware, but
  now pinned rather than merely documented**. The choice this project
  already made (implement `energy_pj`/`latency_ns` fields exactly as
  named, not as `docs/target-format.md`'s prose reading of the spec's
  internally-inconsistent Section 17 worked example) was previously
  enforced by nothing except that prose: a future change to
  `lib/Placement/CostReport.cpp`'s arithmetic could silently reintroduce
  the spec's 1000x error with no test catching it, since the existing
  `cost_report_energy_units_scale_sensibly` test only checks that the
  formatter picks the right *unit suffix* (`uJ`), not the right
  underlying number.

  New test `install_energy_pins_the_pj_as_written_convention`
  (`test/unit/cost_report_test.cpp`) asserts the exact numeric values on
  both ends of `target-format.md`'s own worked example against
  `erbium-8t`'s real `program.energy_pj: 480000`: 8 tiles' install energy
  is exactly `3840000.0` pJ (3.84 uJ, not the spec prose's stated 3.84
  mJ, which is 1000x larger), and amortized over 1e6 inferences that is
  exactly `3.84` pJ/inference (negligible, not the spec's stated 3.84
  nJ/inference, which would make install cost a material 13% of the
  per-inference budget). Mutation-tested by injecting the spec's implied
  1000x factor into `costOfWindow`'s energy arithmetic and confirming
  both assertions go red before reverting.

  This does not resolve which figure is actually correct on real
  hardware -- that still needs measured data, per M5 below -- it only
  makes sure this project's own documented interpretation cannot drift
  by 1000x silently.
- ~~`m != 1` hard-refused in `CIMPartition.cpp` -- v0.1 is matrix-**vector**:
  there is no batching at all, not merely no *dynamic* batching~~ --
  **closed on the compiler side**. A matmul with M > 1 output rows is now
  tiled exactly like the m == 1 case -- weight programming, activation
  staging, mvm, reduce_partial, write-back -- generated once and wrapped in
  a genuine `scf.for` this pass builds itself over the M rows, rather than
  refused. This is deliberately the SAME IR shape
  `test/real-target/check-loop.mlir.in` already proved runs correctly as a
  real binary (a HAND-written loop, from before this pass could generate
  one itself); `cim-placement`'s existing loop hoisting and
  `cim-lower-to-target`'s existing arbitrary-depth loop lowering needed no
  changes to make this land -- both already generalize to whatever loop
  shape reaches them, which is exactly what let PR #5's nested-loop work
  and this land as two separate, independent PRs.

  The one plain (non-`cim`) buffer this pass allocates directly -- the
  K-padding scratch for a ragged activation -- is hoisted by hand once,
  before the generated loop, the same way `cim-lower-to-target` hoists its
  own scratch: a fresh allocation per row would leak M times at runtime
  instead of once.

  **A real, separate, PRE-EXISTING bug found along the way**: batching is
  the first thing to ever feed a NONZERO-offset host memref into
  `cim.copy`'s or `cim.program`'s lowering (every m == 1 activation
  subview was offset 0; every real-target-e2e multi-tile test's weight
  happened to be uniform, so a wrong address read the same value anyway).
  `CIMLowerToTarget.cpp`'s `hostPointer` helper called
  `memref.extract_aligned_pointer_as_index` directly on the source memref
  -- verified against real `--expand-strided-metadata
  --finalize-memref-to-llvm` output that this op returns ONLY the
  underlying allocation's base pointer, lowered to a bare
  `llvm.extractvalue` of the descriptor's aligned-pointer field, and never
  adds the descriptor's separate offset field, static or dynamic. Any
  caller ever handed a genuinely offset host memref got the WRONG address
  silently. Fixed by computing the address as
  `memref.extract_strided_metadata`'s base pointer plus
  `offset * element_width_bytes` by hand. Caught by a new real-target-e2e
  pair (`batch-correct`/`batch-wrong`, `test/real-target/check-batch.mlir.in`)
  built from a genuine multi-row `linalg.matmul_transpose_b` -- not a
  hand-written loop -- through the real `cim-detect`/`cim-partition`/
  `cim-placement`/`cim-lower-to-target` chain: three rows with different
  first elements, checksummed the same way `check-loop.mlir.in` already
  does, so a wrong address reading row 0's bytes on every iteration is a
  wrong, checkable number instead of a lucky match. Mutation-tested by
  reverting the fix and confirming `batch-correct` goes from exit 0 to a
  real `SIGABRT` on the checksum assert, then restoring it.

  New FileCheck coverage: `test/Transforms/cim-partition.mlir`'s
  `multi_row_is_batched_over_an_scf_for` replaces the old
  `multi_row_is_left_alone` refusal test, pinning the generated `scf.for`,
  the dynamic per-row activation/output subviews, and that
  `cim-placement`'s hoisting still applies (verified separately against
  real `cim-opt` output, not just asserted).

  ~~Still open: the ONNX front end still refuses any activation with more
  than one row~~ -- **closed**. `emit.py`'s `emit_module`/`emit_chain_module`
  now accept a real `[M, K]` activation array (a 1-D `[K]` vector still
  works, reshaped to `[1, K]` internally, so the M == 1 output stays
  byte-identical to `test/python/test_numerical_differential.py`'s
  `build_module()` and `test/mlir/pipeline_e2e_test.cpp`'s `buildModule()`
  -- confirmed by every pre-existing emitter test, unchanged). Both of
  `onnx_import.py`'s M != 1 refusal sites (single-layer and a chain's
  first layer) are lifted; `_validate_activation` normalizes a `[K]` or
  real `[M, K]` supplied activation the same way. M threads through a
  chain unchanged, with no chain-specific code needed: `cim.requantize`'s
  own verifier already enforces "must not change shape", so a batched
  layer 0 makes every later layer's matmul and requantize genuinely
  `[M, ...]` too.

  Verified with the same rigor as every other front-end change here, not
  a lighter pass: a new batched differential in `test_onnx_frontend.py`
  (3 shapes x 2 placement settings, independently-sampled per-row values
  so a cross-row mix-up would be a wrong, checkable number) against
  `onnx.reference`, a batched 3-layer chain differential in
  `test_onnx_frontend_chain.py`, and two `test_onnx_emitter.py` cases
  pinning the emitted shape without needing `onnx` installed at all.
  `test_onnx_frontend_refusals.py`'s old `test_refuses_more_than_one_
  output_row` is replaced by `test_accepts_more_than_one_output_row`,
  matching the "prove it doesn't refuse everything" discipline the rest of
  that file already follows. Mutation-tested: forcing `emit_module`'s `m`
  to a hardcoded `1` was caught immediately, by MLIR's own elements-literal
  shape check before ever reaching a numerical comparison.
- ~~Only `linalg.matmul_transpose_b` (weights `[N x K]`, matching
  `cim.mvm`'s output-major convention). A plain `linalg.matmul` needs a
  transpose first~~ -- **closed**. `cim-partition` now accepts a plain
  `linalg.matmul` (weights in `[K x N]`, ONNX's own layout) directly,
  transposing the weight into a fresh `[N x K]` buffer itself -- once, at
  weight-staging time, the same place ragged-edge padding already happens
  -- rather than refusing the candidate. The transpose is a genuine,
  doubly-nested `scf.for` over `memref.load`/`memref.store`, deliberately
  NOT a `linalg` buffer op (`linalg.transpose`/`linalg.fill`): this pass's
  own existing padding logic already uses `linalg.fill` on memrefs, and no
  real-target-e2e binary has ever exercised that through a real compiled
  build (every shipped real-target shape is an exact tile multiple, so
  padding's `needsPadding` branch is always false in practice today) --
  `scf.for` is the one loop-emitting primitive this pass already knew
  lowered correctly through the real `--convert-to-llvm` pipeline (the
  M > 1 batching loop above uses it), so the new real-target-e2e coverage
  a plain-matmul candidate gets tests this new code specifically, rather
  than incidentally depending on a separate, still-unverified gap.

  The ONNX front end does not directly exercise this path today -- it
  already transposes the weight itself in Python at import time and always
  emits `linalg.matmul_transpose_b` (see `onnx_import.py`'s own "THE
  TRANSPOSE" note) -- so this closes a retargetability gap for a different
  future front door: `lib/Transforms/CIMDetect.cpp`'s own comment already
  names "an inline constant (tensor semantics, e.g. from torch-mlir)" as a
  shape a frontend can produce, and the top-level README names plugging in
  below IREE/TVM/XLA as this project's own stated position. A frontend
  that does not pre-transpose no longer needs to.

  Deliberately still out of scope: `linalg.batch_matmul` (a real 3rd batch
  dimension, not just M-row batching, and materially more machinery) stays
  refused, matching `cim-detect`'s own documented "detected, then refused
  with a warning" status for it.

  Verified: `test/Transforms/cim-partition.mlir`'s
  `plain_matmul_is_transposed_and_tiled` replaces the old
  `plain_matmul_layout_is_left_alone` refusal test, pinning the generated
  transpose loop and the tiled `cim.program`/`cim.mvm` sequence that
  follows it. New real-target-e2e pair `plain-matmul-correct`/`-wrong`
  (`test/real-target/check-plain-matmul.mlir.in`) uses a deliberately
  NON-symmetric weight, so a missing or backwards transpose reads entirely
  different values rather than merely reordered ones -- the same
  discipline `test_onnx_frontend.py`'s own transpose tests already use.
  Mutation-tested: swapping the transpose loop's load indices (`weights[i][j]`
  instead of `weights[j][i]`) was caught immediately by the real compiled
  binary's checksum assert, then reverted.
- ~~`scale=1.0`/`zero_point=0` always: v0.1 has no per-layer calibration
  step anywhere in the pipeline~~ -- **partially closed**. `cim.requantize`
  and `cimrt_requantize` were already fully scale-generic (proven at
  `scale=2.0, zp=3, effective_bits=4` in
  `legalize_precision_e2e_test.cpp`); the only thing standing between that
  and a real calibrated model was the ONNX chain bridge's own hard-coded
  `scale == 1.0` refusal. `_validate_bridge` (`onnx_import.py`) now accepts
  any positive scale, threaded from each `QuantizeLinear` node through
  `load_matmul_chain`/`import_model` into `emit_chain_module`'s
  `cim.requantize` text. `zero_point` stays pinned at 0 -- deriving a real
  zero point from calibration data is a separate, harder problem this
  change does not touch, which is why "partially".

  This changes what "matches the oracle" means: `scale=1.0` was exact
  because rounding an already-integer accumulator has no fractional part
  for round-half-to-even (ONNX) and round-half-away-from-zero
  (`cim.requantize`) to disagree about. A real scale reintroduces genuine
  ties, so this is tested as two separate claims rather than one. First,
  an ODD integer scale makes a tie mathematically impossible (`v / s` at a
  half-integer requires `s/2` to itself be an integer, which an odd `s`
  never is), so `test_a_real_calibrated_scale_matches_the_quantized_
  reference` checks an odd-scale chain against `onnx.reference`'s own
  fully-quantized evaluation (not an unquantized oracle -- that stopped
  being valid the moment scale left 1.0) for an exact match, with no
  hedging. Second, `test_a_real_scale_at_an_exact_rounding_tie_documents_
  the_known_divergence` hand-constructs a minimal (1x1, 1x1) chain where
  ONNX and `cim.requantize` provably land on opposite sides of a tie, and
  asserts the divergence explicitly, by exactly 1 -- documenting the
  known rounding-mode difference as a modeled hardware fact rather than
  hiding it, the same way the M2 note above already characterizes
  `scale=1.0` as a special case rather than the general rule.

  Mutation-tested, and this caught a real test-quality gap, not just the
  intended one: the first attempt at the odd-scale test used a small
  scale (3.0) against randomly-generated, full-int8-range weights.
  Hardcoding the emitter's scale back to `1.0` still passed, because at
  that weight magnitude every accumulator saturates to the same +-128
  clamp regardless of which small scale divides it -- the test was
  checking that saturation matches, not that the scale threads through.
  Fixed by picking a scale (199) large enough that this fixture's
  specific accumulators requantize to their exact, unclamped value, and
  by adding an explicit assertion that the scale=199 and scale=1.0
  references actually differ for this fixture -- a permanent guard against
  the same mutation being silently unmasked again by a future change to
  either the weights or the scale.

  A genuine, unrelated bug was found via this feature's own differential
  test, not by inspection: `cim-run` segfaulted on the two-layer chain
  MLIR this scale support emits. AddressSanitizer traced it to
  `Interpreter.cpp`'s `memref.cast` handler --
  `memrefs[o.getResult()] = it->second;` -- a classic self-referential
  `DenseMap` insert: `operator[]` on the left evaluates (and can grow/
  rehash the table) before the right-hand side `it->second` is read,
  invalidating `it` first. Triggered by this particular module's
  allocation count happening to land on a rehash boundary, not by the
  scale value itself -- a latent bug the interpreter has carried since
  `memref.cast` support was added, exposed by the first differential test
  with this exact shape. Fixed by copying the value out to a local
  before the insert (`MemRefValue` is a `shared_ptr` plus small vectors,
  cheap to copy). Every other map-write site in `Interpreter.cpp` was
  audited for the same pattern (`runSubView`, `runLinalgFill`, `runMvm`,
  `runReducePartial`'s partial-sum loop, `runCimCopy`, `runRequantize`)
  and confirmed to either write a freshly-constructed value or never read
  the stale reference again after the write -- this was the only
  instance. Mutation-tested by reverting the fix and confirming the
  ASan build reproduces the original heap-use-after-free before
  restoring it.
- **A single 2-D convolution (`QLinearConv`) is importable, via im2col --
  no dialect, pass, or interpreter change.** The commercial gap-analysis
  review that opened this milestone flagged "Model Generality" as v0.1's
  headline limitation and specifically named convolution support as
  quarters-of-work item, alongside transformer attention and dynamic
  shapes -- true in general, but this project's execution model is not
  general. onnx_import.py's own module docstring already establishes that
  every emitted module bakes ONE inference: cim-run has no mechanism to
  pass data in, so the activation is always a compile-time constant. That
  single fact removes im2col's usual cost (materializing every
  overlapping window redundantly, at every inference) entirely -- it
  happens exactly once, in Python, at import time, before any MLIR
  exists. So a 2-D convolution over a constant activation and a constant
  kernel reduces to exactly the (weight `[N, K]`, activation `[M, K]`)
  shape `emit.emit_module` already emits for a plain matmul: reshape the
  kernel `[Cout, Cin, Kh, Kw] -> [Cout, Cin*Kh*Kw]` (already Cout-major,
  unlike MatMulInteger's own weight -- no transpose needed, see
  `im2col.py`'s own module docstring), im2col the activation into
  `[N*OutH*OutW, Cin*Kh*Kw]` (`python/cim_frontend/im2col.py`, free of any
  `onnx` dependency, same reason `emit.py` is), and emit the same
  `linalg.matmul_transpose_b` `cim-detect`/`cim-partition`/the interpreter
  already fully verify. `N*OutH*OutW` flattens straight into the
  already-existing M > 1 batching machinery above -- no new code needed
  for a batched conv either.

  `QLinearConv`, unlike `MatMulInteger`, always quantizes its own output
  in one node (`x_scale`/`w_scale`/`y_scale`), so `emit.emit_module`
  gained an optional `trailing_requantize` parameter: a single,
  unconditional `cim.requantize` after the layer this function ever
  emits (as opposed to `emit_chain_module`'s 0-or-1-per-BRIDGE one).
  Backward compatible by construction -- `trailing_requantize=None`'s
  output is still byte-identical to `build_module()`. The single scalar
  `cim.requantize` needs is derived as
  `y_scale / (x_scale * w_scale)`, which QLinearConv's own reference
  formula makes equivalent to `round(res * (x_scale * w_scale /
  y_scale))` for `res` an integer accumulator.

  Deliberately refused, each named and reasoned about rather than
  silently approximated (`onnx_import.py`'s `load_qlinear_conv` module
  section header has the full list): grouped/depthwise convolution (each
  group is a genuinely separate matmul, not one reshape), dilation (a
  dilated tap reads a non-contiguous patch, which this im2col's
  contiguous-slice implementation does not express), per-output-channel
  (per-axis) quantization (a single `cim.requantize` call takes exactly
  one scalar scale), a non-zero `x_zero_point`/`w_zero_point`/
  `y_zero_point` (same "needs a per-output bias term this dialect cannot
  express" reason `MatMulInteger`'s own zero points are refused for), a
  bias operand (added directly to the raw int32 accumulator per output
  channel -- there is no cim op for that at this point in the pipeline
  without risking an unverified interaction with `cim-partition`'s own,
  unrelated use of `cim.reduce_partial` for K-tiling, the same discipline
  `Gemm`'s bias is refused for), and any `auto_pad` other than `NOTSET`
  (including `VALID`, despite `VALID` being definitionally just
  `pads=[0, 0, 0, 0]` -- found while testing: `onnx.reference`'s own Conv
  implementation computes `VALID`'s padding with the same shape-dependent
  formula as `SAME_UPPER`/`SAME_LOWER`, and that formula indexes
  `X.shape[i]` for a spatial dimension without skipping the leading N/C
  axes, so it is wrong for any model with `N != 1` or `Cin != 1` -- a bug
  in the oracle, not a reason to special-case `VALID` on this side, since
  `NOTSET` with `pads=[0, 0, 0, 0]` -- this function's own default -- is
  already exactly `VALID`'s defined behavior).

  Verified: `test/python/test_onnx_frontend_conv.py` --
  a strided, padded, non-saturating differential against
  `onnx.reference`'s own quantized `QLinearConv` evaluation; a batched
  (N > 1) case proving im2col's batch flattening composes with the M > 1
  machinery; a full-int8-range 1x1-kernel case at an odd derived scale
  (mathematically guaranteed no rounding tie, the same v = scale*(n+0.5)
  argument the chain bridge's own odd-scale test relies on); a
  hand-constructed exact-tie case documenting the same
  half-away-from-zero-vs-half-to-even divergence the chain bridge's own
  tie test documents; one refusal test per restriction above; and a
  direct, `onnx`-free unit test of `im2col_nchw` against an independent,
  non-im2col-based hand-written convolution loop. Mutation-tested twice:
  inverting the derived-scale formula (`x_scale*w_scale/y_scale` instead
  of `y_scale/(x_scale*w_scale)`) was caught by every correctness test;
  swapping im2col's height/width slice indices was caught by every
  correctness test AND the onnx-free `im2col_nchw` unit test. Also caught
  a real test-quality issue while writing it, not a hypothetical one:
  the first version of the batched test used an even `y_scale=4`, which
  hit a genuine rounding tie at one output element -- fixed by switching
  to the odd `y_scale=5`, the same odd-scale-avoids-ties reasoning used
  throughout this front end.

  Deliberately not in scope for this entry: a *chain* of convolutions (or
  a convolution feeding a matmul), matching how single-layer matmul
  import preceded chained matmul import.
- **Per-channel scale, a real bias, and asymmetric zero points -- also
  reshapes, found by testing against a real model, not by inspection.**
  The convolution entry above was validated only against hand-built
  synthetic fixtures. Importing `squeezenet1.0-12-int8` from the ONNX
  model zoo -- a real, ONNX-Runtime-quantized model -- immediately
  refused on the very first layer, three separate times: a
  per-output-channel `w_scale` (64 distinct values), a real int32 bias,
  and an asymmetric (`x_zero_point=115`, uint8) input. None of these are
  exotic; they are the *default* output of a real quantization
  toolchain. Investigated each by hand-building a probe module and
  running it through a real `cim-opt`/`cim-run` round trip before writing
  any of `onnx_import.py`'s new code, exactly the discipline the plain-
  convolution entry above already used -- and found all three (plus a
  fourth, `y_zero_point`) are expressible with existing, already-verified
  machinery, not a dialect change:
  - **Per-channel scale**: N `cim.requantize` calls, one per output
    channel, each against a `memref.subview` of that column.
    `AnyMemRef` accepts a strided view, and the interpreter's gather/
    scatter already handle strides generically (the same machinery
    `cim-partition`'s own ragged-tile padding relies on).
  - **Bias**: a single `cim.reduce_partial(matmul_output, bias_broadcast)`
    between the matmul and the eventual requantize. `cim.reduce_partial`'s
    verifier constrains operand shape and element type only, never
    producer, and `cim-partition` treats a matmul's `outs` buffer as an
    opaque, pre-allocated destination it fills in place -- erasing the
    matmul op once the tiled sequence has written the real answer back
    into that SAME buffer. A hand-emitted `cim.reduce_partial` reading
    that buffer therefore composes with `cim-partition`'s rewrite with no
    special-casing on either side. This closes the "unverified
    interaction" the plain-convolution entry's bias refusal named as its
    reason -- it turned out to compose cleanly the first time it was
    actually tried.
  - **Asymmetric `x_zero_point`**: this project's "one baked inference"
    contract makes the activation a compile-time constant, so
    `X - x_zero_point` is computed once, in Python, before any MLIR is
    emitted -- exactly like im2col itself. Exact whenever
    `w_zero_point == 0` (still required; see below), and refused, naming
    the actual out-of-range values, if the shifted result does not fit
    signed int8.
  - **Asymmetric `y_zero_point`**: `cim.requantize` already carries a real
    `zero_point : i32` attribute (`legalize_precision_e2e_test.cpp`
    proves it at `zero_point=3`), and `cimrt_requantize`'s order of
    operations -- round, THEN add zero_point, THEN clamp -- already
    matches QLinearConv's own reference formula exactly. The only change
    was to stop hard-coding it to 0.

  Still refused, and for a real (not merely unexplored) reason:
  `w_zero_point != 0` -- the `w_zp * X` cross term in QLinearConv's true
  accumulator is per-ROW (activation-dependent), not a fixed per-channel
  correction the way a bias is, so it is not a reshape of anything that
  already exists. In practice this costs little: real toolchains commonly
  keep weights symmetric even when activations are asymmetric, exactly as
  `squeezenet1.0-12-int8` itself does (whose own `w_zero_point`, checked
  by hand, ships as a length-64 array where every entry happens to be 0 --
  itself a small, real finding: `w_zero_point` needed the same
  scalar-or-per-channel-*shape* handling as `w_scale`, even though its
  *value* must stay entirely zero either way). A non-scalar
  `x_zero_point`/`y_zero_point` also stays refused: per-tensor activation/
  output quantization is the near-universal convention, unlike per-channel
  *weight* scale.

  **Capstone verification, beyond the fixture-based tests below**: the
  real `squeezenet1.0-12-int8` first layer -- its actual per-channel
  weight scales, its actual bias, its actual `x_zero_point=115` -- was
  extracted into a standalone single-node model and run through the real
  `cim-opt`/`cim-run` pipeline. Every one of the 14,400 output elements
  (a 15x15x64 feature map) matched `onnx.reference`'s own evaluation of
  the same real model exactly. The one remaining gap this exposed at the
  time: that layer's `y_zero_point` is declared `uint8`, and
  `cim.requantize`'s clamp is a signed `effective_bits` range that cannot
  represent a uint8 output's full `[0, 255]` span -- refused explicitly
  (not silently wrapped) rather than forced through.

  **Closed, without a dialect change.** `clamp(-128, 127, t - 128) ==
  clamp(0, 255, t) - 128` for every real `t` -- both clamp bounds shift by
  exactly 128, so the shift commutes with clamping exactly, not
  approximately. Requantizing with `(y_zero_point - 128)` instead of the
  declared `y_zero_point` therefore produces EXACTLY
  `(true_uint8_output - 128)` in every element, algebraically, not a
  lucky coincidence at the extremes. The emitted module's own provenance
  header discloses the shift explicitly (`load_qlinear_conv`'s
  `uint8_output_shifted` flag), on the same "the front end's job ends at
  'produce the right numbers'; a documented caller-side transform is not
  a silent one" discipline as the NHWC reshape a conv's own caller
  already has to apply. Verified by a differential test at a real,
  non-edge `y_zero_point` (130, not 0 or 128, so an off-by-one or a
  forgotten shift could not accidentally still pass) against
  `onnx.reference`'s own uint8 output, mutation-tested by perturbing the
  shift constant by one and confirming the comparison catches it. Not
  re-run against the actual `squeezenet1.0-12-int8` file -- this
  session's sandboxed network policy blocks the ONNX model zoo fetch that
  produced the original capstone extraction -- so this is a synthetic
  fixture at realistic parameters, not a second capstone run; said
  plainly rather than implied otherwise.

  **A follow-up self-check, immediately after landing the shift, found
  `analyze.py`'s permissive walker still carried the OLD refusal.**
  `_qlinear_conv_kn` had its own copy of the "y_zero_point is uint8;
  cim.requantize's clamp is SIGNED" check, correct at the time it was
  written, stale the moment `load_qlinear_conv`'s matching refusal was
  lifted above. Left unfixed, `cim-import-onnx --emit-workload` would
  have kept reporting a uint8-output conv layer as `skipped` -- silently
  under-counting a real model's own offloadable layers for a reason that
  had stopped being true, on exactly the kind of real layer
  `squeezenet1.0-12-int8`'s own first layer is. The dtype check is
  removed (shape never depended on it, the same reason dilation needs no
  check there either); the scalar-ness check beside it stays, since a
  non-scalar `y_zero_point` is still genuinely refused by the compile
  path. Pinned by a regression test, mutation-tested by reintroducing the
  stale check and confirming it goes red.

  Verified: `test/python/test_onnx_frontend_conv.py` -- a bias
  differential with three distinct, non-symmetric channel values (so a
  row/column mix-up in the broadcast reads as a loud wrong number), a
  per-channel-scale differential with three deliberately different
  scales, an asymmetric-`x_zero_point` differential (uint8 input,
  zero_point=120), an asymmetric-`y_zero_point` differential, one
  refusal test per restriction above (including the all-zero-but-
  per-channel-shaped `w_zero_point` acceptance found via the real model),
  and a "prove it doesn't refuse everything" test for that exact shape.
  Mutation-tested: inverting the per-channel derived-scale formula,
  flipping the sign of the `x_zero_point` subtraction, reversing the
  bias array's channel order, and disabling the `w_zero_point`-must-be-
  zero check were each caught by the correctness or refusal tests built
  for that specific capability.

- **Cost-model integrity: every declared target field either charges what
  it says or is refused, not silently ignored.** A fresh audit of the
  whole project, run against its own stated philosophy ("every declared
  knob must be real, every executable operation must be cost-charged,
  anything unsupported must be refused loudly"), found the published cost
  report was not honest yet. Five defects, each landed with the test that
  would have caught it:
  - `cim-cost-report`'s JSON hardcoded `num_tiles`,
    `standby_leakage_uw_per_tile`, and `double_buffer_capable` to struct
    defaults instead of the parsed target -- every target was published as
    a leak-free device, silently, because every existing lit test happened
    to use `tiny-4x4.yaml`, whose declared values coincide with the
    defaults. `test/Transforms/cim-cost-report-target-fields.mlir` runs
    against a target where they differ.
  - `cim.copy` was never counted in the static report even though
    `costs.transfer.energy_pj_per_byte` is a **required** target field --
    an entire declared cost class silently zero. Now counted
    (`CostReportUtils.cpp`), with the gap that remains -- `lowerProgram`/
    `lowerMvm`'s implicit weight/activation staging via `cimrt_write`,
    which no `cim.copy` op represents and no IR walk can see -- disclosed
    explicitly (`transfer_bytes_excludes_implicit_staging`,
    `host_to_host_copies`) rather than silently absent, and NOT asserted
    equal to the runtime side in `cost_report_e2e_test.cpp`'s differential,
    with a comment explaining why that assertion would be false advertising:
    the compiled path hoists staging out of loops while the interpreter
    re-stages per op, so the two executors already disagree with each
    other before static-vs-runtime is even asked.
  - `costs.transfer.bandwidth_gbps` -- required, and mandatory for every
    vendor -- charged zero latency; every byte moved took 0 ns. Now charged
    on both the runtime side (`CostAccumulator::recordTransfer`) and the
    static report, reading `gbps` as **gigabytes** per second (`ns = bytes
    / gbps` exactly), pinned by a fixture (`tiny-4x4-bandwidth.yaml`,
    `bandwidth_gbps: 4.0`) chosen specifically because every prior test
    target's `bandwidth_gbps: 1.0` makes the correct reading, the gigabits
    reading, and no conversion at all collapse to the same number -- the
    same "passes for the wrong reason" trap a saturating quantization
    fixture fell into earlier in this project.
  - `tiles.weight_dtype`/`activation_dtype` -- required, unread, and
    silently mis-executed: a target declaring `weight_dtype: i4` opened,
    compiled, and ran with `int8_t` reinterpretation at full declared cost,
    because the passes take element types from the memrefs, never from the
    target. The check belongs in `cimrt_open` (execution), not
    `TargetYAMLParser` (parsing) -- the first attempt in the parser broke
    `test_yaml_differential.py` correctly, because that test compares
    reader **syntax** against PyYAML, and `i4`/`u8`/`i16` are valid
    *spellings* even though nothing executes them yet.
    `tiny-4x4-i4-weights.yaml` must parse but must not open.
  - `cim-detect` declined a candidate with three bare `return;`s and no
    diagnostic at any level, in a file whose own header says convolution
    "must not be silently accepted" -- a dropped op made the entire
    pipeline quiet, offloading nothing with no message, while the very
    next pass warns four different ways for the same kind of decline. Now
    a remark naming the op and the reason, pinned by
    `test/Transforms/cim-detect-remarks.mlir` (which documents two
    FileCheck hazards hit while writing it: `--split-input-file`
    concatenates all sections' diagnostics into one stream, and FileCheck
    scans comment prose too, so naming a directive in an explanation makes
    the explanation a directive).

  Also closed: no CI job ran the ONNX front end's *generated* IR under
  sanitizers -- `mlir-asan` was C++/lit only, and the interpreter's
  `memref.cast` use-after-free (fixed earlier in M2) lived exactly in IR
  shapes only the front end produces. `mlir-asan` now also runs `pytest
  test/python`; a first attempt silently skipped 44 of 180 tests because
  only `cim-opt`/`cim-run` had been built, which is the same defect class
  (a check that quietly does less than it claims) as everything else on
  this list.

  Left open, deliberately, because none of them produces a wrong number
  today: `cim.program`'s `cost_ns`/`cost_pj` attributes are dead IR (zero
  readers); `tiles.persistence` (string) is inert and can contradict
  `tiles.persistent` (bool); the `class:` enum is unread but carries
  neither a disclosure nor a pinning test the way `autonomous_control`
  does.

- **Real-model placement/cost analysis: point the placement engine at a
  real network without executing it.** This project's differentiated
  asset -- Belady tile eviction, the amortization model, the target cost
  schema -- could previously only be driven by five hardcoded synthetic
  workloads (`makeV01Workloads`). The enabling insight: **placement
  analysis needs only weight shapes, never execution.**
  `makeLayeredWorkload` takes nothing but a `vector<uint32_t>` of per-layer
  block counts, and `partitionBlockCount(k, n, tileRows, tileCols)` derives
  each purely from a layer's `[K, N]` -- so a real model can be *analyzed*
  even though large parts of it (MaxPool, Concat, Softmax, a grouped or
  dilated convolution) cannot yet be *compiled*.
  - `python/cim_frontend/analyze.py` -- unlike `import_model`, which
    refuses an entire graph on the first unrecognized op, this walks every
    node and classifies it: an offloadable `MatMulInteger`/`QLinearConv`
    goes in `layers` with its `[k, n]`, anything else goes in `skipped`
    with a stated reason, and the walk never aborts. Reuses
    `onnx_import.py`'s field-level validators (`_zero_point_is_zero`,
    `_positive_weight_scale`, `_weight_zero_point_must_be_zero`, …)
    directly rather than duplicating their logic, so a defect reads as the
    same sentence whether it blocked compilation or was merely skipped
    here. One deliberate, narrow divergence from the strict loaders: a
    grouped or dilated `QLinearConv` is still counted as offloadable,
    because ONNX already stores its weight as `[Cout, Cin/group, Kh, Kw]`
    -- the real per-filter footprint -- so the shape is known and correct
    even though this front end cannot yet *emit* IR for it.
  - `cim-import-onnx --emit-workload model.onnx -o w.json` -- a second,
    unrelated entry point needing no `--input`/`--input-random`: unlike
    compiling, shape-only analysis never needs an activation.
  - `cim-bench analyze --workload-file w.json --target t.yaml` -- maps
    each layer's `[k, n]` through the existing `partitionBlockCount` into
    `makeLayeredWorkload`, then reuses the Belady/LRU/FIFO comparison and
    cost report unchanged. Reads the interchange file with a small,
    dependency-free JSON reader (`lib/Placement/WorkloadJSON.cpp`, same
    rationale as the YAML target reader: `cim-bench` and `cimPlacement`
    build with no LLVM/MLIR toolchain, so no JSON library is available
    either) that refuses -- rather than silently defaults -- a document
    missing the `skipped` field, so "nothing was skipped" can never be
    confused with "the producer forgot to say". `test/unit/
    workload_json_test.cpp` pins the rejection table the same way
    `parser_error_test.cpp` pins the YAML reader's; `test/python/
    test_workload_json_differential.py` runs the real `cim-bench` binary
    against documents Python's own `json` module also parses, including
    `\u` escapes and quote/backslash-bearing names, the same "compare the
    shipped artifact against an implementation written by other people"
    discipline as `test_yaml_differential.py`.
  - The honesty requirement, checked by test, not just stated: every
    report -- Python's JSON and `cim-bench analyze`'s stdout and JSON
    alike -- states in words that the numbers are weight-programming cost
    for the N offloadable layers *only*, not end-to-end inference cost,
    and names what was skipped and why.
  - `test/workloads/small-cnn-workload.json` is a checked-in fixture (a
    hand-built three-conv/two-maxpool graph -- this session's sandboxed
    network policy returns 403 on a raw GitHub content fetch, so it is not
    an actual downloaded model, unlike the convolution feature's
    `squeezenet1.0-12-int8` capstone above) kept current by
    `test/python/test_analyze.py::test_checked_in_workload_fixture_is_current`,
    the same regenerate-and-diff convention `test_onnx_frontend.py` uses
    for `onnx-imported-matmul.mlir`. It lets `test_cim_bench_analyze.py`
    exercise the full Python-front-end-to-C++-placement-engine path,
    including a mutation check that enlarging one real layer's `K`
    increases the placed `programs` count, with no network fetch or
    `onnx` dependency in the C++-only jobs.

- **The three low-severity audit findings left open above, closed the
  same way `autonomous_control` was: disclosure plus a pinning test,
  applied uniformly rather than left as a known gap.** None of these
  produced a wrong number before this — that is what made them
  low-severity — but each was a claim (in a comment or in
  `docs/target-format.md`) that nothing actually checked.
  - **`cim.program`'s `cost_ns`/`cost_pj` are a provenance record, not a
    live cost source, and the comment that set them said the opposite.**
    `lib/Transforms/CIMPartition.cpp` sets them correctly from the target
    file at emission time, but grepping every downstream pass, the
    interpreter, and the runtime for a reader of either attribute finds
    none — `cim-cost-report`, `cim-lower-to-target`'s `lowerProgram`, and
    the interpreter's `runProgram` each independently re-parse the target
    file instead. The comment on the line that sets them claimed the
    opposite ("carries its own cost so later passes can reason about
    reprogramming without a target lookup") — false, corrected in place.
    `test/Transforms/cim-program-cost-attrs-are-unread.mlir` hand-writes a
    `cim.program` with `cost_ns`/`cost_pj` set to `999999999` (a value no
    real `cim-partition` run against `tiny-4x4.yaml` could ever produce)
    and checks `cim-cost-report`'s own numbers are the target's, not the
    IR's — mutation-tested by making the pass actually add the attribute's
    value into its total and confirming the test goes red.
  - **`tiles.persistence` (string) can no longer silently contradict
    `tiles.persistent` (bool).** The string was documentation-only —
    `docs/target-format.md` claimed it "drives the program/mvm cost
    asymmetry", which was also false; `tiles.persistent` is the field
    every pass that branches on volatility actually reads. Rather than
    only disclosing the gap, `TargetYAMLParser.cpp` now refuses a target
    file where both fields are present and disagree (including the
    quieter version of the mistake: `persistence: nonvolatile` with
    `persistent` left unset, which would otherwise silently keep its
    default of `false`). Pinned by a new row in
    `test/unit/parser_error_test.cpp`'s rejection table, and every shipped
    target file already agrees by convention so nothing broke.
  - **The `class:` enum is now pinned the same way `autonomous_control`
    is.** Confirmed still true by grep (zero readers outside the parser
    and `cim-bench dump-target`), but previously unpinned: every existing
    pair of `tiny-*.yaml` fixtures that differ in `class:`
    (`tiny-4x4.yaml` vs `tiny-4x4-4bit.yaml`) also differs in
    `precision.output_effective_bits`, so no test could isolate `class:`
    on its own. `test/targets/tiny-4x4-analog.yaml` is byte-identical to
    `tiny-4x4.yaml` except `class:`, and
    `test/Transforms/cim-class-is-unread.mlir` runs the same module
    through the full eight-pass chain against both and diffs the outputs
    byte for byte — mutation-tested twice: once against a mutation
    (round-robin tile id) that turned out NOT to survive to final output
    because `cim-placement` overwrites it regardless, which is itself a
    useful negative result about where a real regression could hide, and
    once against a mutation in `cim-lower-to-target`'s buffer-space choice
    that does survive to final text, which correctly turned the test red.

- **A self-audit of Phase 1 itself, the same rigor turned on the code that
  did the auditing.** Three findings, all mutation-tested:
  - `analyze.py`'s `_qlinear_conv_kn` called `_int_attr(node, "group", 1)`
    and `_int_list_attr(node, "dilations", [1, 1])` "to check that
    something read-able exists" and discarded the result -- but neither
    helper can actually fail that check: both read the raw protobuf field
    for their expected type and silently return the type's zero value if
    the attribute is stored under a different `AttributeProto` type,
    rather than raising. A call that cannot fail and whose result is
    discarded is not a check; it was dead code dressed up as one. Removed.
  - `analyze_model` walks only `graph.node`, which never visits a nested
    `GraphProto` -- an `If`/`Loop`/`Scan` branch. A `MatMulInteger` inside
    one was previously not merely skipped but genuinely invisible: not in
    `layers`, not in `skipped`, not named anywhere. `_has_subgraph`
    detects any node carrying a `GRAPH`/`GRAPHS`-typed attribute and gives
    it a distinct skip reason saying its subgraph was never entered,
    rather than the generic "not a weight-stationary op" wording that
    would otherwise read as a complete story when it is not one.
  - The C++ `WorkloadJSON` reader accepted `k=0` or `n=0` on a layer (a
    non-negative integer, which the schema's own bounds allow) --
    `partitionBlockCount(0, n, ...)` silently returns 0 blocks for that
    layer, so it would count toward `layers_analyzed` while contributing
    nothing whatsoever to the placement result. Now refused: `k` and `n`
    must both be strictly positive, a matmul with a zero contraction
    dimension or zero output channels not being a degenerate real layer
    but not a layer at all.

- **Dilated convolution (`dilations != (1, 1)`) is now imported, not
  refused.** Broadening convolution support was the next item on the
  "deliberately out of scope" list from Phase 1's own plan. Of the two
  restrictions named there (grouped and dilated), dilation turned out to
  need no real model to motivate lifting it, unlike every other
  convolution capability landed so far (per-channel scale, bias,
  asymmetric zero points) which needed `squeezenet1.0-12-int8` to prove
  necessary: the shape argument alone was sufficient once looked at
  closely. "A dilated kernel tap reads a non-contiguous patch" is true,
  but non-contiguous describes only the SOURCE indices -- the destination
  patch `im2col_nchw` builds per output position is the same dense
  `[C, Kh, Kw]` block either way, and numpy's own strided slicing
  (`start:stop:step`, step = the dilation factor) reads exactly that
  pattern with no new data structure or second pass needed. Grouped
  convolution remains refused: each group is really an independent matmul
  over a slice of channels, which genuinely cannot be expressed as a
  single reshape the way dilation can.

  `im2col_nchw` (`python/cim_frontend/im2col.py`) gained
  `dilation_h`/`dilation_w` parameters (default 1, identical to every
  call site before this landed); `load_qlinear_conv`
  (`onnx_import.py`) validates `dilations` has length 2 and both entries
  positive instead of refusing any value but `(1, 1)`.
  `analyze.py`'s shape-only path needed no change at all: `k`/`n` never
  depended on dilation, so the earlier "grouped or dilated, either way
  the shape is known" framing in its own comments simply lost the
  "dilated" half now that dilation is no longer a compile-path
  restriction to work around.

  Verified: a differential test with deliberately asymmetric
  `dilation_h != dilation_w`, `stride_h != stride_w`, and non-zero
  padding (so a swapped H/W axis anywhere in the plumbing reads as a
  shape error or a wrong number, not a lucky match) against
  `onnx.reference`'s own quantized `QLinearConv` evaluation, plus a
  direct `im2col_nchw` unit test against an independent hand-written
  convolution loop that visits each kernel tap one at a time (not numpy's
  own strided-slice syntax, which the implementation itself uses --
  reusing that trick in the "independent" oracle would really be testing
  the same formula against itself). Both mutation-tested: reverting
  `im2col_nchw`'s dilated sampling back to a contiguous slice turned both
  red; reverting `load_qlinear_conv`'s positivity check turned the
  refusal test red (for the right reason -- it still raised, but a raw
  `ValueError` from deep inside `im2col_nchw` rather than a proper
  `Refusal`, which is exactly what that check exists to prevent).

- **`cim.requantize`'s verifier now rejects a non-integer input or result
  element type, instead of silently accepting IR that two different
  downstream consumers were already independently refusing on their own.**
  Found while auditing `include/cim/Dialect/CIMOps.td`'s own header
  comment claiming every verifier rule from spec Sec. 5.4 is "implemented
  in lib/Dialect/CIM/IR/CIMOps.cpp" -- true for the numbered rules, but
  `RequantizeOp::verify()` carried a leftover: a private
  `memRefElementType(Value)` helper, defined and called
  (`(void)memRefElementType(getInput())`) purely to discard its result --
  the exact "a call that cannot fail and whose result is discarded is not
  a check" pattern already found and removed twice in the Phase 1
  self-audit above, not caught here until now because it lived in the
  dialect's own C++ rather than the front end. Both `Interpreter.cpp`'s
  `runRequantize` and `CIMLowerToTarget.cpp`'s `lowerRequantize` already
  refuse a non-integer input or result element type -- scale/zero_point/
  effective_bits describe an integer quantization and neither has any
  other accumulator representation to read or produce -- but each does so
  independently, with its own wording, only once the module reaches that
  specific pass. `test/Dialect/CIM/invalid.mlir` had exactly this gap
  documented two cases above as precedent (`!cim.tile`'s own
  element-type-must-be-integer check, added "until the verifier took the
  argument and never looked at it"): the same class of bug, in the same
  file, left unfixed on a different op.

  Fixed directly in `RequantizeOp::verify()`: refuse before the
  `effective_bits`-vs-result-width check even runs, matching the
  interpreter's own message ("input and result element types must both be
  integers"). The dead `memRefElementType` helper is removed rather than
  reused -- the new check only needs `dyn_cast<IntegerType>` on types
  already recovered as `MemRefType`s two lines above it, so the extra
  indirection had no purpose to begin with.

  Mutation-tested, and more decisively than most guards in this project:
  removing the check does not merely let a malformed module through
  silently -- it segfaults `cim-opt` outright. With the integer check
  gone, the `effective_bits > outElem.getWidth()` comparison right below
  it (itself simplified from a conditional to unconditional once integer-
  ness is guaranteed) calls `IntegerType::getWidth()` on a null
  `IntegerType` for the new `f32`-result negative test, crashing before
  any diagnostic prints. Reverting restores both negative-test passes and
  a clean `cim-opt` run. This is a property of the mutated code (the
  refactor deliberately trades the old conditional guard for reliance on
  the earlier early-return), not a claim that a pre-existing crash shipped
  in `main` -- but it is a sharp demonstration that the new check is load-
  bearing, not cosmetic.

  Two new `test/Dialect/CIM/invalid.mlir` cases (float result, float
  input) pin this; `test/Dialect/CIM/requantize.mlir`'s existing valid
  case already used integer types throughout and needed no change.

- **`cim-partition`'s `lowerLinalgMatmulCandidate` no longer declares two
  `Type` locals (`i8`, `i32`) it never uses.** Found continuing the same
  dead-code sweep that turned up the `cim.requantize` gap above: `git log
  -L` traces both back to `0da2a5c`, the very first real implementation
  of this pass, replacing a `TODO(spec Sec.6, Pass 2)` placeholder --
  eighteen commits ago, never referenced since. Unlike `memRefElementType`
  above, these were not masking a missing check: the only place a
  `weight_dtype`/`activation_dtype` mismatch can legitimately be caught is
  `cimrt_open` (closed properly in Phase 0's defect D fix, with its own
  commit explaining why partition time is too early to own that refusal
  -- this pass takes element types from the memrefs it is given, never
  from the target file). `cim_build_flags` already compiles this file
  under `-Wall -Wextra -Werror`, and the `(void)i8; (void)i32;` lines
  exist specifically to suppress the `-Wunused-variable` that would
  otherwise fire -- so this was a deliberate, tracked silencing of dead
  code, not an overlooked warning. No behavior change; no new test, since
  there is nothing to pin (removing an unused local cannot regress
  anything the existing suite exercises). Verified by full rebuild (zero
  new warnings), the full lit/ctest/pytest suite, clang-tidy (clang-18
  tree, zero findings), and cppcheck (zero findings).

- **Chain a `QLinearConv` into `MatMulInteger` layers: the first real step
  of "full CNN compilation," the multi-layer counterpart to a single
  convolution.** Full CNN compilation splits into two genuinely
  different-sized problems: chaining conv layers (extends the proven
  `MatMulInteger` chain precedent) and executing non-matmul host ops
  (MaxPool/Softmax/etc., which needs real execution semantics this
  codebase does not have anywhere yet -- neither the interpreter nor the
  `real-target-e2e` `mlir-opt` pass list has any linalg-lowering/
  bufferization stage). This entry is the first, tractable half only.

  Staged from a plan built on three rounds of hands-on research (three
  parallel codebase explorations, then a design pass that hand-verified
  its core recommendation against the real, checked-in `cim-opt`/`cim-run`
  toolchain rather than just reading code -- see that research's own
  finding, quoted below). Landed as the smallest real capability step:
  the convolution is always chain layer 0, so its activation is still the
  literal graph input and `im2col_nchw` still runs once in Python exactly
  as it does for a standalone conv -- everything after that reuses the
  already-proven `MatMulInteger` chain bridge (`emit_chain_module`)
  completely unchanged. Chaining a SECOND convolution is a materially
  larger, separate problem, not started here: the next layer's activation
  would only exist as an MLIR value once the first layer's compiled code
  actually runs, and `lib/Interpreter/Interpreter.cpp`'s `execute()` is a
  closed `TypeSwitch` that cannot even execute the index arithmetic
  (`arith.addi`/`muli`, `scf.if`) a naive loop-based im2col would need --
  confirmed by compiling a probe module through the real binaries and
  reading the exact failure (`error: operation not supported by the cim
  interpreter: memref.expand_shape`). A follow-up design exists for that
  harder half (unroll the spatial gather over `Kh*Kw` kernel taps, each
  one static non-unit-stride `memref.subview`/`memref.copy`, no `scf.for`
  and no arithmetic -- `memref.subview` already supports a static
  non-unit stride generically) but is not implemented.

  **A genuine correctness gap found during implementation, not anticipated
  by the design.** The original plan assumed a `QLinearConv`'s raw ONNX
  output could feed a `MatMulInteger` node directly, mirroring how
  `MatMulInteger`-to-`MatMulInteger` bridges need no reshape. Confirmed by
  hand to be wrong: ONNX's own `MatMulInteger` reference kernel does
  N-D numpy-broadcast matmul rather than requiring 2-D operands, so a
  graph wiring a conv's raw `[N, Cout, OutH, OutW]` output straight into a
  `MatMulInteger` does not error -- it silently contracts whatever axes
  happen to align (e.g. `OutW` against `Cout`, if they happen to be
  equal), a genuinely different, wrong function, not a shape error. This
  is precisely the "confident wrong number" failure class this project
  refuses to ship, caught only by testing the actual `onnx.reference`
  behavior rather than assuming ONNX semantics from the spec's prose.
  Fixed by requiring an explicit `Transpose(perm=[0, 2, 3, 1]) ->
  Reshape([M, Cout])` bridge in the graph -- not merely tolerated, refused
  if absent or wrong -- making the `.onnx` file itself a faithful,
  standards-compliant statement of the same function this compiler emits.
  Confirmed exact by hand: the full `QLinearConv -> Transpose -> Reshape
  -> MatMulInteger` graph's own `onnx.reference` evaluation matches the
  conv's raw NCHW output manually transposed/reshaped and matrix-
  multiplied, to the element.

  `python/cim_frontend/onnx_import.py` gains `load_conv_matmul_chain`
  (validates the conv, the `Transpose`/`Reshape` bridge, and any further
  `MatMulInteger` layers via the existing `_validate_bridge`) and a
  factored-out `_conv_geometry` helper (group/dilations/auto_pad/
  kernel_shape/pads/strides), now shared by `load_qlinear_conv` too rather
  than duplicated -- two copies of that validation drifting apart is
  exactly the failure class analyze.py's own uint8-`y_zero_point`
  regression was (this same M4 section, Phase 1 hardening entry), so one
  copy, called from both places, instead of a second one that could go
  stale relative to the first. `import_model`'s dispatch gains a
  `conv_count >= 1 and matmul_count >= 1` branch (matching the standalone
  conv branch's own `conv_count >= 1`, not `== 1` -- so a graph with too
  many convolutions still reaches the right loader's own refusal, not a
  wrong one via a dispatch gap that would have missed it entirely).

  Deliberately narrower than a standalone conv or a standalone chain,
  because `emit_chain_module`'s bridge is one unconditional scalar
  `cim.requantize` with `zero_point` hardcoded at 0: a chained conv
  requires `y_zero_point == 0` (declared `int8`, not `uint8` -- there is
  no caller downstream of a chain to apply the standalone path's
  documented `+128` shift before the next matmul reads the value), a
  single scalar `w_scale` (no per-channel), and no bias. Lifting any of
  these needs `emit_chain_module` itself to grow a per-channel/bias-aware
  bridge, not just the loader -- not attempted here, on the same "don't
  invent capability a real model hasn't forced" discipline this section's
  own convolution entries have followed throughout.

  Verified against the real compiled pipeline (`cim-opt`/`cim-run`, not
  Python math alone): a single-matmul differential, a two-matmul
  differential with a strided/padded conv and a real (non-1.0) bridge
  scale, a batched (N > 1) differential, and an anti-vacuity check that a
  perturbation in the MATMUL layer (not the conv) is actually caught.
  Mutation-tested, and one of those mutations found a second, independent
  bug -- in the TEST, not the implementation: disabling the "conv output
  must be read by exactly one Transpose" check did not turn
  `test_refuses_a_missing_transpose_reshape_bridge` red, because a LATER
  check (the `perm` validation) also happens to print the literal word
  "Transpose" in its own message while actually describing a mislabeled
  `MatMulInteger` node -- so the test's original bare `match="Transpose"`
  passed for the wrong reason, exactly the "a check that fails for the
  wrong reason is still a bug" principle this project applies to its own
  compiler elsewhere, just this time caught in the test suite itself.
  Fixed by matching the specific guard's own wording
  (`"not read by exactly one Transpose"`) instead of a substring any
  nearby message could satisfy. The perm check and the `y_zero_point`
  check were also mutation-tested against real compiled numbers, not just
  "does the test fail": disabling either one compiles and runs cleanly --
  no crash, no diagnostic -- and produces a completely different result
  from `onnx.reference`, confirmed by hand for both.

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
