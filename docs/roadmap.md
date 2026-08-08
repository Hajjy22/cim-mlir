# Roadmap

Milestones from the v0.1 spec (Section 13). Each has a public artifact —
nothing counts until it is public.

**Current state.** M0, M1 and M3 are done. All eight lowering passes are real,
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
  than discovered partway through: straight-line code only (any cim op
  nested inside a region -- an `scf.for` body, exactly what
  `cim-placement`'s own loop hoisting produces -- is refused with a
  diagnostic, since a buffer this pass allocated would either leak every
  iteration or need a hoisting analysis of its own, neither designed
  here), and `cim.reduce_partial` is refused outright rather than
  mislowered (multi-tile K-reduction needs its own buffer-lifetime story
  this pass does not have -- multiple partial-sum buffers alive at once,
  reduced into one). `cim.requantize`, originally refused for the same
  stated reason, is now lowered (see the composition-hardening entry
  above): it turned out to have no such buffer-lifetime problem in the
  straight-line, single-tile slice, since `cim-legalize-precision` makes
  it a terminal accumulator's SOLE consumer -- one producer, one consumer,
  exactly like `cim.mvm`'s own staged activation.

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
  `cim-partition` output.

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
- [ ] End-to-end: an ONNX INT8 matmul compiles and produces numerically
  correct output vs. PyTorch. Nothing yet connects the compiled IR to the
  simulator, so there is no numerical check across the whole pipeline.

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
- `cim-lower-to-target` beyond its v0.1 straight-line, single-tile slice:
  lowering `cim.reduce_partial` (multiple partial-sum buffers alive at
  once, reduced into one -- its own real buffer-lifetime story), and
  lowering a cim op inside an `scf.for` (needs a real buffer-lifetime
  story across loop iterations, not just within one straight-line block)
  -- see the Pass 7 entry above for exactly what is and is not covered
  today. `cim.requantize` lowering is done (composition-hardening entry
  above).
- `cimrt_requantize` accounted in the cost model: the target schema's
  `costs:` section needs a requantize/readout entry before
  `cimrt_profile_stop`/`cim-cost-report` can count it (currently zero-cost,
  a known simplification, not a silent omission -- see the Pass 7 entry
  above).

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
