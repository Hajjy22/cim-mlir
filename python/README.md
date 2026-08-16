# The ONNX front end

Reads an ONNX model and emits the MLIR the `cim` pipeline consumes. This
is the front door the rest of the project was written behind: everything
else here starts from MLIR that this repository wrote, either by hand in
the FileCheck tests or from a string builder in the differential tests.

## Install

```sh
pip install ./python           # cim-import-onnx CLI only
pip install './python[onnx]'   # + the onnx package the CLI actually needs at runtime
```

This installs a `cim-import-onnx` console script, so the pipe below no
longer needs `PYTHONPATH`:

```sh
cim-import-onnx model.onnx --input act.npy \
  | cim-opt - --cim-detect \
      --cim-partition=target-yaml=targets/erbium-8t.yaml \
      --cim-placement=target-yaml=targets/erbium-8t.yaml \
  | cim-run --target-yaml=targets/erbium-8t.yaml --profile -
```

Without installing, the equivalent is:

```sh
PYTHONPATH=python python3 -m cim_frontend model.onnx --input act.npy \
  | cim-opt - --cim-detect \
      --cim-partition=target-yaml=targets/erbium-8t.yaml \
      --cim-placement=target-yaml=targets/erbium-8t.yaml \
  | cim-run --target-yaml=targets/erbium-8t.yaml --profile -
```

`python/pyproject.toml` packages only this directory (`cim_frontend`);
it deliberately does not depend on anything under `lib/`, `runtime/`, or
`tools/`, since those need an LLVM/MLIR toolchain this package has no
reason to require. The `.github/workflows/ci.yml` `packaging` job installs
it fresh with `pip` and runs the console script for real (not
`python3 -m`, and not through the `test/python` suites' `sys.path` hack)
so this stays true rather than merely documented.

## Why Python, and why not in `tools/`

The output is *text*, consumed by the `cim-opt` binary over a pipe, so
there is no in-memory MLIR handoff a C++ importer would preserve. Against
that, a C++ importer costs real things:

- `libonnx` + protobuf become a hard dependency of the **build**, not of
  an optional test. The `core`, `core-asan` and `core-valgrind` CI jobs
  configure with `-DCIM_ENABLE_MLIR=OFF` and build in seconds; the
  README's claim that the core has no heavy dependencies would need an
  asterisk.
- `.github/workflows/ci.yml`'s `static-analysis` job runs
  `run-clang-tidy-18 -warnings-as-errors='*'` over all of
  `lib|runtime|tools/**/*.cpp`. Protobuf-generated sources under that flag
  means either suppressions or narrowing the glob, and narrowing the glob
  weakens a gate that currently covers everything.

`onnx`'s canonical API is Python, and torch-mlir's own ONNX importer is
Python for these same reasons.

It lives in `python/` rather than `tools/` because every subdirectory of
`tools/` is a CMake target producing a binary in `<build>/bin`, which is
also how `test/python/conftest.py`'s `find_tool()` locates tools. A script
that is not built, needs no MLIR, and cannot be found that way does not
belong there.

## What it accepts

A graph consisting of one or more `MatMulInteger` nodes and nothing else
(chains are bridged as described below), where each layer's operands
satisfy:

- the weight operand (B) is a constant initializer,
- both operands are `int8` (not `uint8`),
- zero points are absent or zero,
- operands are rank 2 (the activation may have one row or many — see
  "Batching" below),
- every dimension is a known constant,
- the model targets opset 10 or later.

`MatMulInteger` is the accepted op because it *is* the v0.1 contract in
one node: int8 A, int8 B, int32 Y — "INT8 in, wider integer accumulator
out". A single, standalone `QLinearConv` node is also accepted, as a
second entry point — see "A single 2-D convolution" below.

## Chained layers

A single `MatMulInteger` is the whole v0.1 contract, but it is also the
one shape where `cim-placement` has nothing to reuse *across* — real
cross-layer eviction, the `mlp-3layer` benchmark shape, needs more than
one weight matrix competing for the tile budget. A graph of two or more
`MatMulInteger` nodes is imported as a chain, provided consecutive layers
are bridged by exactly one pattern: `Cast(to=float32) ->
QuantizeLinear(scale, zero_point=0, int8 output)`. Anything else between
layers — a non-positive scale, a missing zero point, any other op — is
refused.

That specific bridge, and no other, is what keeps the numerical claim
checkable. Each bridge's scale becomes the `scale` operand of a real
`cim.requantize(scale=<scale>, zero_point=0, effective_bits=8)` between the
two compiled matmuls — `cim.requantize` and its runtime implementation are
fully scale-generic, so any positive scale is accepted, not just `1.0`.

What differs is which oracle a real scale can be checked against.
`cim.requantize` rounds half-away-from-zero; ONNX's `QuantizeLinear` rounds
half-to-even — two modes that diverge only at a tie, a value ending in
exactly `.5`. At `scale=1.0`, both sides compute `round(v / 1.0)` on a `v`
that is already an integer (an accumulator), so there is never a
fractional part and therefore never a tie either side could round
differently — the compiled chain matches an *unquantized* ONNX oracle
exactly. At a real (non-1.0) scale, ties are possible in general, but an
**odd integer scale makes one mathematically impossible**: `v / s` lands on
a half-integer only if `s / 2` is itself an integer, which an odd `s`
never is. So an odd-scale chain still matches exactly, checked against
`onnx.reference`'s own *quantized* evaluation instead of an unquantized
one (see `test_a_real_calibrated_scale_matches_the_quantized_reference`).
An even scale can hit a genuine tie, where the two rounding modes really do
disagree by construction, not by bug — `test_a_real_scale_at_an_exact_
rounding_tie_documents_the_known_divergence` pins that divergence
explicitly rather than hiding it. This was checked against a real
`cim-opt`/`cim-run` round trip by hand before the graph-walking code that
finds this pattern was written — see `onnx_import.py`'s own module
docstring for the exact numbers.

## Batching (M > 1)

The activation may have more than one row. `cim-partition` tiles a real
M-row matmul the same way it tiles a single row: the same weight
programming, activation staging, `cim.mvm`, and write-back, generated
once and wrapped in a real `scf.for` it builds itself over the rows,
rather than refused (`docs/roadmap.md`'s M4 entry). This threads through a
chain unchanged too — `cim.requantize` never changes shape, so a batched
layer 0 makes every later layer genuinely `[M, ...]` as well, with no
per-layer bookkeeping in this front end.

Pass a real `[M, K]` array via `--input` (a `.npy` or `.json` file) to use
it; `--input-random` still synthesizes a single row only, since it exists
for quick smoke-testing rather than as the batching entry point.

## A single 2-D convolution (`QLinearConv`)

A graph consisting of exactly one `QLinearConv` node (and no
`MatMulInteger`) is imported too — via im2col, entirely in Python, with no
new MLIR op and no `cim-detect`/`cim-partition`/interpreter change. This
works because of the same fact "The activation is required" (below)
already establishes: every emitted module bakes ONE inference, so the
activation is always a compile-time constant. im2col is normally
considered expensive because it materializes every overlapping window
redundantly, at every inference — that cost does not apply here, since
there is no per-inference cost in this project's execution model at all.
A 2-D convolution over a constant activation and a constant kernel is
exactly a matmul in disguise: the kernel reshapes `[Cout, Cin, Kh, Kw] ->
[Cout, Cin*Kh*Kw]` (no transpose needed — see `im2col.py`'s own module
docstring for why this is different from `MatMulInteger`'s weight), the
activation im2cols into `[N*OutH*OutW, Cin*Kh*Kw]`, and the result is
handed to the exact same emitter this front end already runs a full
verification gate against for a plain matmul. `N*OutH*OutW` flattens
straight into the M > 1 batching machinery above, so a batched convolution
needs no separate support either.

Unlike `MatMulInteger`, `QLinearConv` always quantizes its own output in
one node, so the emitted module ends in a `cim.requantize` even for this
single "layer" — the scale it needs is derived as
`y_scale / (x_scale * w_scale)`, matching `QLinearConv`'s own reference
semantics for an integer accumulator.

Only a plain (non-grouped), explicit-padding convolution is accepted — see
`onnx_import.py`'s `load_qlinear_conv` module section header for the full
list of what is refused and why (each is a real scope boundary, not an
oversight; the refusal table below has an entry for the exact-padding
requirement, which is the one surprise: `auto_pad=VALID` is refused too,
despite being definitionally just zero padding, because `onnx.reference`'s
own `Conv` implementation computes it with a padding formula that is wrong
for any model with more than one image or input channel — an oracle bug
this front end sidesteps by refusing it, rather than a limitation of its
own). A chain of convolutions, or a convolution feeding a matmul, is not
supported — only a single, standalone `QLinearConv`, matching how
single-layer `MatMulInteger` import preceded chained import.

**Dilation (`dilations != (1, 1)`) is accepted** — unlike group, it changes
only which source pixels a kernel tap reads, not the shape of anything:
the destination patch `im2col_nchw` builds per output position is the same
dense block either way, so numpy's own strided slicing (step = the
dilation factor) reads the pattern directly with no new data structure or
second pass. See `im2col.py`'s own "DILATION IS A SAMPLING PATTERN" note.
Grouped convolution remains refused: each group is really an independent
matmul over a slice of channels, which genuinely cannot be expressed as a
single reshape the way dilation can.

**Per-channel scale, a real bias, and asymmetric zero points are also
accepted** — found necessary, not merely nice-to-have, by importing a real
model (`squeezenet1.0-12-int8` from the ONNX model zoo): every layer of a
real, ONNX-Runtime-quantized model uses a per-output-channel `w_scale`, a
real int32 bias, and an asymmetric `x_zero_point`, none of which the first
version of this front end accepted. All three (plus a real
`y_zero_point`) turned out to be expressible with existing,
already-verified machinery rather than a dialect change:

- **Per-channel `w_scale`**: N `cim.requantize` calls, one per output
  channel, each against a `memref.subview` of that column.
- **A bias operand**: one `cim.reduce_partial(matmul_output,
  bias_broadcast)` between the matmul and the eventual requantize —
  composes cleanly with `cim-partition`'s own rewrite because that pass
  treats a matmul's output buffer as an opaque destination it fills in
  place, regardless of what reads it afterward.
- **A non-zero `x_zero_point`**: subtracted from the constant activation
  in Python before im2col ever runs — exact whenever `w_zero_point == 0`
  (still required), refused if the shifted result does not fit signed
  int8.
- **A non-zero `y_zero_point`**: `cim.requantize` already had a real
  `zero_point` parameter; this front end just stopped hard-coding it to
  0.

Still refused: a non-zero `w_zero_point` (the correction term it would
need is per-row, activation-dependent — not a fixed per-channel bias, so
it is not a reshape of anything above), and a non-scalar `x_zero_point`/
`y_zero_point` (per-tensor activation/output quantization is the
near-universal convention, unlike per-channel weight scale).
`squeezenet1.0-12-int8`'s own first layer was extracted standalone and run
through the real compiled pipeline as a capstone check: all 14,400 output
elements matched `onnx.reference` exactly, using its actual per-channel
scales, actual bias, and actual asymmetric input — the one thing that
layer could not be imported with unmodified is its declared `uint8`
output (`y_zero_point` dtype), since `cim.requantize`'s clamp is a signed
range and cannot represent uint8's full span; a real, still-open gap, not
a re-hidden version of the zero-point restriction above.

## What it refuses, and why refusing is the point

It refuses rather than guesses, and the reason is sharper than usual: the
failure mode of guessing here is not a crash. Dropping an op the importer
does not recognise emits a module that compiles cleanly, runs, and
computes a *different function than the model*. A confident wrong number
is worse than a refusal.

| Refused | Because |
|---|---|
| `Gemm` | carries a bias operand `C`; the dialect has no bias op, so supporting it means dropping or fusing `C` — both silently change the arithmetic |
| `MatMul` + `QuantizeLinear`/`DequantizeLinear` | float arithmetic; offloading it means choosing scales, i.e. calibration, which `cim-legalize-precision` also refuses to invent |
| `QLinearMatMul` | quantized int8 output reintroduces rounding-mode divergence |
| any other op in the graph | see above — silently dropping it changes the computed function |
| non-constant weight | weight-stationary hardware has nothing to make resident |
| two constant operands | `cim-detect` needs exactly one, and silently declines otherwise |
| `uint8` operands | `cim.mvm` and the simulator are signed; reading a `uint8` 200 as `int8` computes with −56 |
| non-zero zero points | needs a per-output bias term the dialect cannot express |
| symbolic/dynamic dimensions | weights are materialized as dense literals |
| a chain bridge with a non-positive scale | `cim.requantize` requires a positive scale; a real positive scale is accepted (see above) |
| a chain bridge with no or non-zero zero point | needs an explicit int8 zero point of 0, both to fix the output dtype at int8 and to stay symmetric |
| a value read by more than one node along a chain bridge | a DAG needs buffer-liveness reasoning this importer does not do; only a strictly linear chain is imported |
| a grouped/depthwise `QLinearConv` (`group != 1`) | each group is really an independent matmul over a slice of channels, not one reshape |
| a `QLinearConv` with a non-positive `dilations` entry | not a sampling pattern any real accelerator (or numpy's own strided slicing) can express; a positive dilation is accepted (see above) |
| a `QLinearConv` with a non-zero `w_zero_point` | its correction term is per-row (activation-dependent), not a fixed per-channel bias, so it is not one of the reshapes above |
| a `QLinearConv` with a non-scalar `x_zero_point` or `y_zero_point` | per-tensor activation/output quantization is the near-universal convention, unlike per-channel weight scale |
| a `QLinearConv` with a uint8 `y_zero_point` | `cim.requantize`'s clamp is a signed `effective_bits` range; it cannot represent a uint8 output's full `[0, 255]` span |
| a `QLinearConv` with `auto_pad` other than `NOTSET` | `SAME_UPPER`/`SAME_LOWER`/`VALID` all need shape-dependent padding math this front end does not replicate — including `VALID`, whose own oracle (`onnx.reference`) computes it with a formula that is wrong for `N != 1` or `Cin != 1` |
| more than one `QLinearConv` node | only a single, standalone convolution is imported; not a chain of them, and not one feeding a matmul |

## The activation is required

The emitted module is `func.func @main()` with no arguments — `cim-run`
takes no runtime inputs — so the module is **one baked inference** and the
activation is a constant.

There is deliberately no default. A zero activation would not merely be
arbitrary, it would be *actively harmful*: `0 @ W == 0` for every `W`, so
every downstream numerical check would pass against any weight matrix at
all, including one that was never transposed. Pass `--input` (a `.npy` or
`.json`), or `--input-random <seed>` to synthesize one explicitly — which
is recorded in the emitted module's provenance header.

## The transpose

ONNX stores the weight as `[K, N]`. `cim.mvm` indexes `W[n][k]`, so the
importer emits `Bᵀ`. This is the one genuinely dangerous thing in the
front end: getting it backwards produces structurally perfect IR and
wrong numbers, and on a square weight it is completely silent.
`test/python/test_onnx_frontend.py` guards it three ways — rectangular
shapes throughout so a missing transpose is a loud shape error, a direct
assertion on the emitted literal, and a mutation check proving that
assertion has teeth.

## Which passes to run downstream

`cim-detect`, `cim-partition` and `cim-placement` reproduce the model's
arithmetic exactly. The full eight-pass chain additionally runs
`cim-legalize-precision`, which inserts a `cim.requantize` clamping every
accumulator to the target's `precision.output_effective_bits` (8 in every
shipped target). That is a real modeled hardware effect, not a bug — but
it means the result stops matching an *unquantized* ONNX oracle as soon as
any accumulator leaves `[-128, 127]`.

## Analyzing a real model, without compiling it

Everything above answers "does this model compile and run correctly".
`--emit-workload` answers a different, narrower question: "how much
weight-residency pressure does this real network put on a given chip" —
without needing the whole graph to be offloadable, and without needing an
activation at all.

```sh
cim-import-onnx real_model.onnx --emit-workload -o workload.json
cim-bench analyze --target erbium-8t --workload-file workload.json --out results.json
```

The insight this rests on: **weight-residency cost is a function of
shape, not execution.** `cim-bench`'s placement engine only ever needs a
layer's `[K, N]` — never its values, and never an activation — so a real
network's `MaxPool`, `Concat`, `Softmax`, or even a grouped convolution
this front end cannot yet *compile* does not block *analysis* of the
layers around it.

`--emit-workload` (`cim_frontend.analyze`) walks the graph permissively —
unlike every path above, it never refuses the whole model. Every node
becomes either an offloadable layer (its `[k, n]` goes in the output's
`layers`) or a skip (its op type and a stated reason go in `skipped`), and
the walk always completes. `cim-bench analyze` reads that JSON with a
small, dependency-free reader (`lib/Placement/WorkloadJSON.cpp`, same
LLVM-free rationale as the target-YAML reader), maps each layer's `[k, n]`
through the existing `partitionBlockCount` into a real placement problem,
and runs the same Belady/LRU/FIFO comparison and cost report `cim-bench
run`'s built-in workloads use.

**This is not end-to-end inference cost, and every output says so.** Both
the JSON `--emit-workload` produces and `cim-bench analyze`'s own output
(stdout and JSON) state in words how many layers were analyzed and how
many other ops were skipped, and name the reason for each skip — a number
that silently represented only part of a network is exactly the kind of
confident-but-partial result this project refuses to publish anywhere
else.

## Tests

- `test/python/test_onnx_emitter.py` — needs no ONNX and no build; pins
  the emitted IR shape against the pipeline's own module builder.
- `test/python/test_onnx_frontend.py` — the end-to-end differential
  against `onnx.reference` (the spec's own implementation) and
  `onnxruntime`, including a batched (M > 1) case with independently
  sampled per-row values.
- `test/python/test_onnx_frontend_refusals.py` — one test per refusal
  above, plus a case proving it does not refuse everything.
- `test/python/test_onnx_frontend_chain.py` — the multi-layer counterpart:
  a 3-layer differential (the `mlp-3layer` shape), a batched 3-layer
  differential, a case that actually saturates the bridge's clamp,
  placement invariance across layers, the chain-specific refusals (bad
  scale, bad zero point, fan-out), a real (odd, non-1.0) calibrated scale
  matched against `onnx.reference`'s quantized evaluation, and a hand-built
  exact-tie case documenting the known rounding-mode divergence at an even
  scale.
- `test/python/test_onnx_frontend_conv.py` — the convolution counterpart:
  a strided/padded differential against `onnx.reference`'s own quantized
  `QLinearConv` evaluation, a batched (N > 1) case proving im2col's batch
  flattening composes with M > 1, a full-int8-range 1x1-kernel case at an
  odd derived scale (guaranteed no rounding tie), a hand-built exact-tie
  case documenting the known rounding-mode divergence, a per-channel-scale
  differential, a bias differential, asymmetric `x_zero_point` and
  `y_zero_point` differentials, one refusal test per convolution-specific
  row above (including the all-zero-but-per-channel-shaped
  `w_zero_point` a real model ships), and a direct, `onnx`-free unit test
  of `im2col_nchw` against an independent hand-written convolution loop.
  Also validated, outside CI, against `squeezenet1.0-12-int8` (ONNX model
  zoo) itself — see `docs/roadmap.md`'s M4 entry for the exact result.
- `test/Transforms/onnx-imported-matmul.mlir` — importer output checked
  in, so the `mlir` CI job guards the shape without the ONNX packages.
- `test/python/test_analyze.py` — `analyze_model`'s own contract: a valid
  matmul/conv is offloaded not skipped, an unrecognized op is skipped with
  a named reason rather than crashing the walk, a grouped convolution is
  still offloaded (shape-only analysis, unlike compilation), the honesty
  `note` field, and a regenerate-and-diff check on the checked-in
  `test/workloads/small-cnn-workload.json` fixture.
- `test/python/test_workload_json_differential.py` — the C++
  `WorkloadJSON` reader vs Python's own `json` module on the same file,
  including `\u` escapes and quote-bearing names.
- `test/python/test_cim_bench_analyze.py` — `cim-bench analyze` end to
  end against the checked-in fixture, including a mutation check that
  enlarging one real layer's `K` moves the placed program count.
- `test/unit/workload_json_test.cpp` — the JSON reader's own rejection
  table, same convention as `parser_error_test.cpp`'s for the YAML reader.

Install the optional dependencies with:

```sh
pip install -r test/python/requirements-onnx.txt
```

Without them these tests skip; they never fail.
