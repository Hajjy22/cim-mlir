# Roadmap

Milestones from the v0.1 spec. Each needs a public artifact — nothing counts until it is
public.

**M0–M3 are done.** A real `.onnx` file compiles and runs, all eight passes are
implemented and compose end to end on one module, and the placement result is measured
against a known optimum rather than asserted.

> The detailed decision log that used to live in this file — every design tradeoff, probe,
> and mutation test, layer by layer — is in the git history. This page is the status.

## M0 — Environment

- [x] Repository scaffold.
- [x] `cim-opt` builds against MLIR 18. No from-source LLVM build needed: Debian/Ubuntu
      ship `libmlir-18-dev`, which turned a 45–90 minute cold CI build into an apt install.
- [ ] Toy tutorial chapters 1–7.
- [ ] Erbium emulator running locally.

## M1 — Dialect skeleton (complete)

- [x] `cim` dialect in ODS: types, attributes, ten ops.
- [x] Verifiers for spec rules 1 and 4, plus shape, geometry and accumulator checks. Rules
      2, 3 and 5 need information a single op cannot see, and say so in the source rather
      than being silently skipped.
- [x] A FileCheck round-trip test per op, plus `invalid.mlir` proving the verifiers really
      do reject bad IR.
- [x] CI builds and tests both layers.

## M2 — Functional correctness (complete)

All eight passes are implemented. See the table in the top-level
[`README.md`](../README.md) for what each one does.

- [x] Passes 1–8, each with FileCheck structural coverage and, where the pass changes
      values, a numerical differential against an independent reference.
- [x] A functional simulator and `cimrt` runtime with a real INT8 MVM.
- [x] The passes **compose**: `test/Transforms/cim-pipeline-full.mlir` chains all eight in
      spec order on one module, down to real `cimrt_*` calls with no `cim` ops left.
      Getting there required closing three genuine composition breaks between passes that
      were individually correct.
- [x] An ONNX INT8 matmul compiles and produces numerically correct results, checked for
      exact int32 equality against ONNX's own reference implementation.

Verification discipline worth stating once, because it applies throughout: a pass that does
not change values (`cim-schedule`, `cim-insert-transfers`) is checked by an *invariance*
test — run with and without it, the numbers must be identical — plus structural tests that
catch what invariance cannot. A pass that *does* change values
(`cim-legalize-precision`) is checked against an independently written reference. Every
guard is mutation-tested: break it deliberately, confirm the suite goes red for the right
reason, revert.

Separately, a full read-through of `cimrt` and the interpreter found seven real defects — a
use-after-free when a buffer outlived its device, an allocation failure that threw a C++
exception across the `extern "C"` boundary, misleading status codes, a profiling window
that never windowed anything, an unreachable open path, an undocumented filesystem trust
boundary, and sub-byte element types computing zero-byte allocations via `bitWidth / 8`.
All seven are fixed, each with a regression test that failed against the unfixed code
first.

### Known limits of cim-partition

Anything genuinely out of scope — a `linalg.conv_2d`-shaped op, a batched matmul's third
dimension, dynamic shapes — is refused with a warning and the `linalg` op left intact, so
the module stays correct and is simply not offloaded.

A 2-D convolution over a *constant* activation is a different claim, and it is closed one
layer up: the ONNX front end reshapes it into the same `linalg.matmul_transpose_b` this
pass already accepts, entirely in Python, before `cim-detect` ever runs.

Three limits that used to be here are now closed:

- **Weight layout.** A plain `linalg.matmul` (`[K, N]`, ONNX's own layout) is accepted and
  transposed by this pass, not refused.
- **Single output row.** A matmul with M > 1 rows is tiled the same way one row is,
  wrapped in an `scf.for` the pass builds itself.
- **Exact tile multiples.** A ragged N or K is zero-padded up to the next multiple
  (`memref.alloc` + `linalg.fill` + `memref.copy`), and the write-back crops each block
  back down to its real rows. Padding can only contribute `0 * x`, so it cannot change an
  answer — checked both structurally and numerically against the unpadded reference.

Buffer lifetime management is out of scope for v0.1 across this pass: its scratch buffers
are not freed, and that is consistent, not an oversight in one place.

## M3 — The placement pass (complete)

- [x] Belady/MIN eviction implemented and unit-tested, with LRU and FIFO baselines. A
      property test checks it against exhaustive search over ~4700 instances.
- [x] An analytical cost model over the target file's cost table.
- [x] `cim-bench` running five workloads end to end, plus plot scripts.
- [x] `cim-placement` rewriting real IR: recovering the use sequence from emitted
      `cim.program` ops, solving it, erasing programming it has proven redundant, and
      assigning tile ids from the solution rather than round-robin.
- [x] Hoisting `cim.program` out of an `scf.for` when its tile is provably untouched for a
      whole iteration and the trip count is a compile-time constant proven positive. This
      reproduces the headline claim on real IR: a model that fits reprograms **once**,
      however many inferences run.
- [x] **The spill gap, measured rather than disclaimed.** On `mm-spill-2x` the pass emits
      `7 + 9*T` programs — 9007 at 1000 inferences, against the optimum's 8538. That is
      **5.5%**, not the 16000 a vaguer phrasing could be read as. `cim-placement` runs the
      flattened solve itself on the real IR and `cim-cost-report` publishes both numbers
      and the gap, so it stays measured.
- [x] **That per-body optimum is deliberate, not accidental.**
      `cim::computeSteadyStatePlacement` pins the `tiles-1` most-used weights and streams
      the rest through the remaining tile. Proved exactly optimal by exhaustive search when
      every weight in a body is used once (`cim-partition`'s own shape); a validated
      heuristic — never worse than pinning nothing — when a weight repeats. `cim-placement`
      takes it whenever it beats the ordinary per-block solve.
- [x] `cim-cost-report` (Pass 8) walking the final placed IR and emitting the project's
      publishable numbers, reusing `cim-bench`'s own JSON format rather than a second cost
      path that could drift from it. Every op is weighted by its enclosing loops' trip
      counts, because a hoisted `cim.program` executes once while one left inside executes
      trip-count times. Checked against `cim-run --profile`'s real counters: predicted and
      executed must agree exactly.
- [x] A volatile-vs-non-volatile amortization comparison showing the two curves cross.

Why `7 + 9*T` is the floor for a loop: a weight no op in the body programs holds a tile
permanently, at most `tiles-1` weights can do that, and everything else is reprogrammed
every iteration. The unrestricted solve beats it only by varying its per-iteration program
count, and a fixed loop body cannot emit a varying number of ops.

## M4 — Generalization (in progress)

Closed so far:

| Area | What landed |
|---|---|
| **Retargetability** | Persistence is the axis that actually mattered — every earlier test target declared `persistent: true`, so `cim.program`'s own attribute had never been checked as `false` past the parser. Now backed by a real compiled-binary pair. No pass branches on `class:` at all, so that tag was never the risk. |
| **Loop nesting** | `cim-lower-to-target` lowers cim ops inside plain `scf.for` nests at any depth. Every buffer it creates is hoisted to before the *outermost* loop and reused, then freed once after it — otherwise each iteration leaks one buffer's worth. `scf.if` and loop-carried `iter_args` are still refused with a diagnostic. |
| **Cost-model integrity** | Every declared target field now either charges what it claims or is pinned by a test as deliberately unread. `cimrt_requantize`, `cim.reduce_partial` and `cim.reduce_max` are each charged at their own entry. |
| **`partial_sum_in_place`** | Read by both the compiled lowering and the interpreter: one accumulator for the whole chain instead of one per step. |
| **Batching (M > 1)** | `cim-partition`'s `m != 1` refusal lifted; a multi-row matmul becomes a real `scf.for`. |
| **Weight layout** | A plain `linalg.matmul` is transposed rather than refused. |
| **Real scales** | `cim-legalize-precision` and the chain bridge accept any positive scale, not only 1.0. |
| **Convolution** | A single `QLinearConv` imports via im2col, then chains: conv→matmul, conv→conv, and a conv stem→matmul head. Per-channel scale, real bias, asymmetric and `uint8` zero points all landed because a real model (`squeezenet1.0-12-int8`) needed them — its first layer matched `onnx.reference` on all 14,400 output elements. |
| **`cim.reduce_max`** | The first executable non-matmul primitive: a signed elementwise max, with its own runtime call, interpreter path, and target cost entry. |
| **`MaxPool`** | Imports between two convolutions, emitted as strided taps folded by one `cim.reduce_max`, padded with `-128`. |
| **`lowerReduceMax`** | The compiled real-target lowering for `cim.reduce_max`, closing the last dialect asymmetry — every op now has both an interpreter path and a compiled path. Non-contiguous operands are **materialized** into a fresh contiguous buffer before staging, because `byteSizeOf`/`hostPointer` both assume contiguity and the front end's pooling taps are exactly the opposite. Verified against a real compiled binary on the real 4-tap window shape before it was automated. |
| **Real-model analysis** | `--emit-workload` plus `cim-bench analyze` point the placement engine at a real network's shapes without needing the whole graph to be offloadable. |
| **Packaging** | `python/pyproject.toml`, installed and exercised fresh by CI. |
| **Units** | The pJ-as-written convention is pinned by a test on a non-unit fixture, so the alternative readings actually differ. |
| **`max_in_place`** | `cimrt_reduce_max_inplace` plus its own capability flag — the identical fold `partial_sum_in_place` gives `cim.reduce_partial`, given to `cim.reduce_max` through a **separate** flag rather than reusing that one, since a compare-and-select is a different datapath element from an adder and a target may support one fold and not the other. `test/targets/tiny-4x4-max-inplace.yaml` declares the opposite combination from every other test target (`max_in_place: true`, `partial_sum_in_place: false`) specifically to prove the two flags are read independently, not one bit doing double duty. |
| **`MaxPool` dilation** | `dilations != (1, 1)` accepted on a pool the same as on a convolution — this was a refusal to lift, not a feature to build: `emit_conv_chain_module`'s own pooling gather already threaded dilation through byte-for-byte identically to the conv-to-conv gather, dead code until the loader's refusal came out. Confirmed against `onnx.reference` first — unlike `strides == 1`, dilation is not an oracle gap. |
| **Pooling composed with the conv-to-matmul bridge** | `load_conv_pool_chain_matmul_chain`: a `MaxPool` may now sit between a conv chain's own last layer and its first matmul layer, not only strictly between two convs. Composed from existing pieces — `_discover_conv_pool_chain`'s interior detection plus `load_conv_chain_matmul_chain`'s own Transpose/Reshape bridge — with one new arithmetic claim: the bridge's own `M` must reflect the pooled spatial size, not the raw last conv's. `emit_conv_chain_module` needed one real fix, not just wiring: its own `pool_params` validation pre-emptively refused this exact bridge index under a stale assumption ("the next layer must be a conv to gather from the pool") that predated a matmul tail ever being able to follow a pool at all; the emission code itself already handled it correctly. Mutation-tested by disabling the new M-shrink step, which fails clean as a `Refusal` (wrong Reshape target shape), not a wrong number. |
| **Grouped/depthwise convolution** | `QLinearConv` with `group > 1` decomposes into `group` independent `linalg.matmul_transpose_b` ops, each with its own correctly-sized weight and activation, concatenated column-wise into one output buffer — not a single block-diagonal matmul with cross-group entries zeroed, because that would make `cim-partition` program real (mostly-zero) tiles and `cim.mvm` burn real cycles multiplying by zero, exactly the dishonest efficiency this project's own cost model exists to catch. Each group is independently recognized by `cim-detect`, confirmed by direct probing. Verified against independent NumPy convolution loops before writing any front-end code (both a real grouped case and a fully depthwise case matched `onnx.reference` exactly — no oracle gap, unlike `MaxPool`'s `stride == 1`). |
| **Grouped/depthwise convolution feeding a matmul chain** | `load_conv_matmul_chain`'s own conv layer is always the chain's own layer 0, so its grouped output only has to be a flat `[M, Cout]` buffer to feed `emit_chain_module`'s existing bridge unchanged — `emit.emit_grouped_conv_matmul_chain_module` produces exactly that (the grouped conv's own G-way matmul + column-concatenation, then `emit_chain_module`'s per-layer loop verbatim). Every OTHER chain loader — a conv chained directly into further convs, or composed with a pooling bridge — still refuses `group != 1`: there, a grouped layer's own per-group Cin/Cout would have to thread through gather/reshape machinery built for one dense channel count, real unstarted work, not a flag flip. |
| **`Relu`, on a matmul chain's bridge** | Needed no new dialect capability: every interior bridge already requires `zero_point == 0`, so a dequantized value's sign matches its quantized value's, and `Relu(dequant(q)) >= 0 iff q >= 0` — Relu on the quantized int8 activation is exactly `max(q, 0)`, computed by the same `cim.reduce_max` MaxPool already uses, against a fresh zero-filled buffer instead of a second pooling tap. Confirmed against `onnx.reference` directly (Relu on a signed int8 tensor computes `max(x, 0)` byte for byte — no oracle gap) and against a real `cim-opt`/`cim-run` round trip of the exact `matmul -> requantize -> reduce_max(x, 0) -> matmul` shape before any front-end code was written. `_strip_optional_relu` recognizes a `Relu` only in the one position that matters — directly producing a later layer's own input — so a `Relu` anywhere else in the graph still falls into the ordinary "unrecognized op" refusal; a position check, not a type allow-list. |
| **`Relu`, on a `QLinearConv` chain, its conv-to-matmul bridge, and composed with pooling** | The identical `reduce_max(x, 0)` composition, now threaded through `emit_conv_chain_module`'s own `relu_bridges` parameter (one shared bridge loop covers every position: interior conv-conv, the conv-to-matmul bridge, and a matmul-tail bridge, for free). `_discover_conv_chain` and `_discover_conv_pool_chain` each generalized in place — not a third parallel discovery function — to recognize a `Relu` directly between two convs the same way `_strip_optional_relu` does on the matmul side, with `_discover_conv_pool_chain` additionally recognizing `Relu` immediately followed by `MaxPool` at the same bridge (ONNX's own `Conv -> Relu -> MaxPool -> Conv` order; the reverse order is not recognized, and does not occur in real ONNX graphs). Hand-verified before any test was written: a hand-built `Conv -> Relu -> MaxPool -> Conv` module matched an independent computation through both the interpreted and compiled `cim-opt`/`cim-run` paths, and `relu(pool(x)) == pool(relu(x))` was checked numerically (both are max-chains with 0 as one more operand) before relying on emitting Relu ahead of a pool in ONNX's own order rather than after it. Every "unused node" and "wrong position" guard is mutation-tested. Explicitly still out of scope: a `Relu` on a chain's own *final* layer output (no bridge exists there — would need a step inside `_emit_bias_and_requantize`'s own tail) and a `Relu` directly before a conv-pool chain's own trailing (pre-matmul-bridge) pool. |
| **`Relu`, directly before `load_conv_chain_matmul_chain`'s own conv-to-matmul bridge** | Row 149 above wired `relu_bridges` through `emit_conv_chain_module` for this bridge position "for free," but the loader's own discovery never actually walked onto it: `_discover_conv_chain` stops once every conv is chained, with no idea a matmul bridge follows the chain's own last conv at all, so a `Relu` there fell into the ordinary "not read by exactly one Transpose" refusal (confirmed by direct testing, not assumed from the docstring, which had explicitly scoped it out). Closed the same way `load_conv_pool_chain_matmul_chain` already closes its own analogous trailing-pool gap: one extra check, right before the Transpose/Reshape validation, hopping across a single `Relu` reader (with its own "exactly one reader" fan-out guard) before requiring a `Transpose`. No emitter change needed — every bridge `emit_conv_chain_module` emits, including this one, already has `zero_point == 0`. The "every `Relu` present must be used" check moved to after this new hop (it used to run immediately after `_discover_conv_chain` returned, before this bridge position existed to check). Mutation-tested: dropping the new `relu_after` entry, and dropping the fan-out check, both fail existing tests for the right reason. Still out of scope: the identical position on `load_conv_pool_chain_matmul_chain`'s own trailing-pool bridge, and a `Relu` between matmul layers. |
| **`Relu`, on a conv chain's own final layer output** | The one Relu position with no bridge at all -- `_discover_conv_chain` never looks past the chain's own last conv, so `load_conv_chain` recognizes it via its own graph-output check instead: the graph's declared output is either the last conv's own "Y" directly, or that "Y" read by exactly one Relu whose own output IS the declared output. Threaded into a new `emit_conv_chain_module` `final_relu` flag, applied inside `_emit_bias_and_requantize`'s own tail, right after its requantize step. This is `max(q, 0)`, same as every other Relu position -- **not** `max(q, zero_point)`, even though the final layer's own `y_zero_point` may be non-zero (every interior bridge instead requires it to be zero). Confirmed directly against `onnx.reference` before writing any code: `Relu` has no `zero_point`/scale attributes at all, and is defined as the literal elementwise `max(x, 0)` regardless of what `x` represents (`Relu([-5, -1, 0, 3, 10]) == [0, 0, 0, 3, 10]` on a signed int8 tensor, byte for byte) -- the `zero_point == 0` requirement everywhere else exists only so `max(q, 0)` in the quantized domain equals `max(dequant(q), 0)` in the real one, not because `Relu` itself needs it. The final layer carries no such requirement, and `Relu`'s own definition does not change. A Relu correctly named as the graph's own declared output, but ALSO read by something else, is refused (there is nothing past it for the emitted module to compute) rather than silently ignored. |
| **`Relu`, on a *conv-pool* chain's own final layer output** | `load_conv_chain`'s identical acceptance above, extended to `load_conv_pool_chain`'s own equivalent graph-output check -- the same final-layer position, now reachable on a chain that also has pooling somewhere else in it. Pooling directly after the chain's own last layer remains unsupported either way (this loader's own long-standing restriction, unrelated to Relu): a trailing `MaxPool` there is still caught by the pre-existing "every MaxPool present must be used" check, which runs before the final-layer check ever sees it, since `_discover_conv_pool_chain` never places a post-chain MaxPool into `pool_after` in the first place. Mutation-tested the same way as `load_conv_chain`'s own final-layer detection. |
| **`Relu`, directly before `load_conv_pool_chain_matmul_chain`'s own trailing pool (or matmul bridge)** | The one bridge position that loader's own module header explicitly named as unrecognized: `Conv -> Relu -> MaxPool -> Transpose`, or (with no pool at that bridge) `Conv -> Relu -> Transpose` directly, mirroring `load_conv_chain_matmul_chain`'s own identical bridge-Relu acceptance. `_discover_conv_pool_chain` never looks past the chain's own last conv, so the trailing Relu-hop is detected here, immediately before the existing trailing-pool check -- an optional Relu, THEN an optional pool, at the same bridge, both new positions this loader's own trailing-position code did not previously consider. No emitter change needed -- every bridge `emit_conv_chain_module` emits, including this one, already has `zero_point == 0`. Mutation-tested: dropping the new `relu_after` assignment made both new positive tests fail via the "unused Relu" refusal rather than a silently wrong number. |
| **SSA-naming hardening across the chain emitters** | `emit_chain_module`, `emit_conv_chain_module`, and `emit_grouped_conv_matmul_chain_module` used to name each weight's SSA value after its ONNX tensor name (`%{sym} = memref.get_global @{sym}`), while also emitting fixed internal names elsewhere in the same module (`%a0`, `%w{i}`). A model whose weight tensor happened to be named `a0` or `w0` -- an entirely ordinary ONNX name -- emitted that SSA value twice, and `cim-opt` rejected the module with "redefinition of SSA value": a loud failure, not a wrong number, but one an unremarkable tensor name should never trigger. Found by code review, reproduced end to end through `import_model`, and fixed by generating every SSA name positionally, independent of any ONNX name -- exactly what `emit_module` always did. ONNX names still label the `memref.global` symbols, which is where they belong; the three symbol lists `emit_grouped_conv_matmul_chain_module` takes now also share one explicit namespace check, since a direct (non-`import_model`) caller could otherwise hit the same class of collision on the global symbols themselves. |

Still open:

- Real (or better-estimated) numbers for `targets/generic-digital-cim.yaml` — it still
  carries placeholders.
- `cim-legalize-precision` with real `effective_bits` modeling; there is still no
  calibration step anywhere in the pipeline to derive a scale from.
- `scf.if` and loop-carried `iter_args` in `cim-lower-to-target`.
- `Relu` between matmul layers — every other Relu position, including both
  final-layer cases (unpooled and pooled), the plain and pooled conv-to-matmul
  bridges, and the pooled trailing bridge, now accepted.
- `GlobalAveragePool` — investigated, not a front-end composition task like `Relu` or
  grouped conv turned out to be. `QLinearGlobalAveragePool`'s real ONNX input is the
  previous layer's own already-quantized int8 `Y`; summing many int8 values into an int32
  accumulator has no existing-op path, because `cim.reduce_partial`'s verifier
  (`verifyElementwiseReduction`) requires every operand AND the result to share one integer
  element type — it does not widen. The interpreter has no `arith.extsi`/`linalg.generic`
  capability to widen a buffer elementwise either. This needs a real new dialect capability
  (a widening reduce, or an explicit widen op) — the same scope `cim.reduce_max` itself
  needed before any front-end code could use it — not a wiring change.
- `Concat`.
- Grouped/depthwise convolution composed into a conv-to-conv chain or a pooling bridge —
  landed standalone and feeding a matmul chain so far.

## M5 — Community and real hardware (future)

- Real Erbium-8T hardware backend — `runtime/src/erbium/erbium_backend.cpp` currently stubs
  every entry point with `CIMRT_ERR_NO_DEVICE`.
- Upstream contribution to AiNEKKO/Erbium.
- Conference talk submission.

## M6 — Decision point (future)

Assess against the spec's own signals — external engagement, MLIR learning curve,
competitive landscape — and choose: raise/recruit, take a role in the space, or continue as
an open-source maintainer.

## Deviations from the spec

Two places where the spec could not be implemented literally. Both were found by compiling
it, not by reading it.

**Pass naming.** The spec names the seventh pass `--cim-lower-to-<target>`, implying one
flag per target. MLIR pass registration requires a single fixed name, so it is one
`cim-lower-to-target` pass parameterized by `-target-yaml=<path>`.

**`cim.tile_alloc` syntax.** The spec prints only the result type, which is unparseable:
`!cim.device` carries a target-name parameter, so the operand type cannot be inferred. It
prints as a functional type instead:

```mlir
%t = cim.tile_alloc %dev {id = 0 : i64} : (!cim.device<"erbium-8t">) -> !cim.tile<256x256xi8>
```
