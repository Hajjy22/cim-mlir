# The ONNX front end

Reads an ONNX model and emits the MLIR the `cim` pipeline consumes. This is the front door
the rest of the project was written behind.

## Install

```sh
pip install ./python           # cim-import-onnx CLI only
pip install './python[onnx]'   # + the onnx package it needs at runtime
```

```sh
cim-import-onnx model.onnx --input act.npy \
  | cim-opt - --cim-detect \
      --cim-partition=target-yaml=targets/erbium-8t.yaml \
      --cim-placement=target-yaml=targets/erbium-8t.yaml \
  | cim-run --target-yaml=targets/erbium-8t.yaml --profile -
```

Without installing, replace the first line with
`PYTHONPATH=python python3 -m cim_frontend model.onnx --input act.npy`.

The package covers `cim_frontend` only — it deliberately depends on nothing under `lib/`,
`runtime/`, or `tools/`, which would drag in an LLVM/MLIR toolchain. CI installs it fresh
with `pip` and runs the console script for real, so this stays true rather than merely
documented.

**Why Python:** the output is *text*, piped into `cim-opt`, so a C++ importer would
preserve no in-memory handoff — but it would make `libonnx` + protobuf a hard dependency of
the build and put protobuf-generated sources under a `-warnings-as-errors` clang-tidy gate.
`onnx`'s canonical API is Python, and torch-mlir's own ONNX importer is Python for the same
reasons.

## What it accepts

| Graph shape | Notes |
|---|---|
| One `MatMulInteger` | The v0.1 contract in one node: int8 A, int8 B, int32 Y |
| A chain of `MatMulInteger` | Bridged by `Cast(float32) -> QuantizeLinear(scale, zero_point=0)`, optionally followed by `Relu` directly on that bridge's own output |
| One `QLinearConv` | Via im2col, entirely in Python — no new MLIR op |
| `QLinearConv` -> `MatMulInteger` chain | Bridged by `Transpose(perm=[0,2,3,1]) -> Reshape([M, Cout])` |
| A chain of `QLinearConv` | Directly connected, no bridge node — both `X` and `Y` are already `[N,C,H,W]`, optionally with `Relu` (and `Relu` then `MaxPool`) directly between two layers, or directly on the chain's own final layer output (with or without pooling elsewhere in the chain) |
| A conv chain -> matmul chain | The realistic full-CNN shape, optionally with `Relu` between convs, or directly before the conv-to-matmul bridge (with or without a pool there too) |
| `MaxPool` between two convs | The first non-matmul op this front end executes, not just analyzes |
| `MaxPool` composed with the conv->matmul bridge | A pool between the chain's own last conv and its first matmul layer — max is scale-invariant, so this needs no bridge of its own beyond moving the Transpose/Reshape's own source |
| A grouped or depthwise `QLinearConv` (`group > 1`), standalone or feeding a `MatMulInteger` chain | `group` independent im2col matmuls, one per channel slice, concatenated into one output — not yet chained into further convs or composed with pooling |

Per layer, operands must be: constant int8 weights, rank 2 (or 4 for conv), all dimensions
statically known, opset 10 or later.

The activation may have any number of rows (M > 1). `cim-partition` tiles a multi-row
matmul the same way it tiles one row, wrapping the work in an `scf.for` it builds itself.

### Convolution details

A 2-D convolution over a *constant* activation is a matmul in disguise: the kernel
reshapes to `[Cout, Cin*Kh*Kw]`, the activation im2cols into `[N*OutH*OutW, Cin*Kh*Kw]`,
and the existing matmul emitter takes it from there. im2col is normally expensive because
it materializes overlapping windows at every inference — that cost does not apply here,
since every emitted module bakes exactly one inference.

Accepted: dilation, per-channel `w_scale`, a real int32 bias, asymmetric zero points, and
a `uint8` output. All four were found necessary by importing a real model
(`squeezenet1.0-12-int8`), and all four turned out to be expressible with existing
machinery — per-channel scale as N `cim.requantize` calls over subviews, the bias as a
`cim.reduce_partial`, `x_zero_point` folded into the constant activation in Python, and
uint8 output via the exact identity `clamp(-128,127, t-128) == clamp(0,255,t) - 128`.

That uint8 shift is **disclosed in the emitted module's provenance header**: the printed
result is `(true value - 128)` and the caller must add 128 back.

For a conv layer that is not the graph's own output, the bridge is one scalar
`cim.requantize` with `zero_point = 0`, so interior layers require `y_zero_point == 0`, a
scalar `w_scale`, and no bias.

### `MaxPool` details

Plain ONNX `MaxPool` carries no scale or zero point, because max is scale-invariant. So a
pooling layer needs no requantize: the previous conv's bridge already produced int8,
`MaxPool` folds int8 to int8, and the next conv reads it directly.

Emitted as `Kh * Kw` static strided `memref.subview` taps folded by one `cim.reduce_max`.
That op compares **signed**, matching what ONNX `MaxPool` on int8 actually computes
(`max(5, -1) == 5`, not `-1`'s raw byte).

**Dilation (`dilations != (1, 1)`) is accepted**, the same as on a convolution — the
emitted gather already threaded dilation through its tap-offset formula byte-for-byte
identically to the conv-to-conv gather (same `output_size()` call, same `th * dilation_h`
pattern), so this was a refusal to lift rather than a feature to build. Confirmed against
`onnx.reference` first: unlike `strides == 1`, dilation is not an oracle gap — a real
dilated, strided, padded integer `MaxPool` evaluates cleanly there, and the `-128` pad
value still loses to a real value at any dilation.

The pad value is **`-128`**, not `0` — a convolution's correct pad value is wrong here,
since `0` can beat a real negative activation it should lose to. Confirmed against three
independent implementations.

Pooling compiles all the way to a real-target binary, not just `cim-run`:
`cim-lower-to-target` materializes each non-contiguous pooling tap into a fresh contiguous
buffer before staging it.

### `Relu` details

`Relu` is accepted directly on a bridge's own output — `Cast -> QuantizeLinear -> Relu ->`
the next layer — needing no new dialect capability. Every interior bridge already requires
`zero_point == 0`, so the dequantized value `scale * q` (`scale > 0`) has the same sign as
the quantized value `q` itself: `Relu(scale * q) >= 0` iff `q >= 0`, so Relu on the
quantized activation is exactly `max(q, 0)`. That is computed by the same `cim.reduce_max`
`MaxPool` already uses, against a fresh zero-filled buffer instead of a second pooling tap
— reusing an existing op rather than adding one.

Confirmed against `onnx.reference` directly (Relu on a signed int8 tensor computes
`max(x, 0)` byte for byte — no oracle gap) and against a real `cim-opt`/`cim-run` round
trip of the exact `matmul -> requantize -> reduce_max(x, 0) -> matmul` shape before any
front-end code was written.

Only the exact bridge position is recognized: on a `MatMulInteger` chain, a `Relu` node
must be the immediate producer of a later layer's own input; on a `QLinearConv` chain, it
must sit directly between two layers' own `X`/`Y` (optionally followed immediately by a
`MaxPool`, matching ONNX's own `Conv -> Relu -> MaxPool -> Conv` node order — the reverse
order is not recognized), directly on the conv-to-matmul bridge (with or without pooling
there too, and with or without an interior pool elsewhere in the chain), between a conv
chain's own last layer and the `Transpose` that starts it, or directly on a conv chain's
own final layer output (with or without pooling elsewhere in the chain) — the graph's own
declared output IS that Relu's own output, reading the last conv's "Y" directly, with
nothing else reading it. A `Relu` anywhere else
in the graph — off on an unrelated branch, between `Cast` and `QuantizeLinear`, after a
pool instead of before it, duplicated — is not silently accepted; it falls into the same
"graph also contains ..." refusal as any other unrecognized op. This is a position check,
not a type allow-list.

The final-layer position is `max(q, 0)`, exactly like every other position — **not**
`max(q, zero_point)`, even though the final layer's own `y_zero_point` may be anything at
all (every interior bridge instead requires it to be exactly zero). Confirmed directly
against `onnx.reference`: `Relu` has no scale/zero_point attributes whatsoever, and is
defined as literal elementwise `max(x, 0)` regardless of what `x` represents. The
`zero_point == 0` requirement everywhere else exists only so `max(q, 0)` in the quantized
domain equals `max(dequant(q), 0)` in the real one — not because `Relu` itself needs it.

Still refused: pooling directly after a conv chain's own last layer (regardless of Relu —
a long-standing, unrelated restriction of its own); and a `Relu` between matmul layers.

### Grouped/depthwise convolution details

`QLinearConv` with `group > 1` splits both input and output channels into `group` slices;
ONNX stores the weight already sliced, `[Cout, Cin/group, Kh, Kw]`, so each group's own
kernel carries only its own `Cin/group` input channels — no cin-axis slicing needed, only
a cout-axis split of the weight and a cin-axis split of the activation. Depthwise is the
special case `group == Cin == Cout` (`Cin/group == 1`).

This is emitted as `group` **independent** `linalg.matmul_transpose_b` ops, each over its
own correctly-sized weight and im2col'd activation slice, concatenated column-wise into
one `[M, Cout]` output buffer via `memref.subview` copies — the same per-column pattern
`per_channel_requantize` already uses elsewhere in this file. Each group is then
independently recognized as its own `cim-detect` candidate, confirmed by direct probing.

The alternative — one `[Cout, Cin, Kh, Kw]` dense matmul with cross-group weight entries
forced to zero — would reuse the plain matmul emitter unchanged, but was rejected: it would
make `cim-partition` program real (mostly-zero) tiles and `cim.mvm` burn real cycles
multiplying by zero, exactly the dishonest efficiency this project's own cost model exists
to make impossible to hide.

Verified against independent NumPy convolution loops before any front-end code was
written — both a real grouped case and a fully depthwise case matched `onnx.reference`
exactly, so there is no oracle gap here (unlike `MaxPool`'s `strides == 1`).

A grouped conv may also feed a `MatMulInteger` chain: `load_conv_matmul_chain`'s own conv
layer is always the chain's own layer 0, so its grouped output only has to be the same flat
`[M, Cout]` buffer an ungrouped conv's bridge already expects —
`emit.emit_grouped_conv_matmul_chain_module` produces exactly that (the same G-way matmul +
concatenation above, feeding straight into `emit_chain_module`'s own bridge-then-matmul
loop, unchanged).

Chained into further convs, or composed with pooling, still refuses `group != 1`, for now:
there, a grouped layer's own per-group Cin/Cout would have to thread through gather/reshape
machinery built for one dense channel count, since the grouped layer is not always the
chain's own layer 0 the way it is here.

## What it refuses, and why refusing is the point

The failure mode of guessing here is not a crash. Dropping an unrecognized op emits a
module that compiles cleanly, runs, and computes a *different function than the model*. A
confident wrong number is worse than a refusal.

| Refused | Because |
|---|---|
| `Gemm` | its bias operand `C` has no dialect op; dropping or fusing it changes the arithmetic |
| `MatMul` + `Quantize/DequantizeLinear` | float arithmetic; offloading it means inventing calibration scales |
| `QLinearMatMul` | quantized int8 output reintroduces rounding-mode divergence |
| any other op in the graph | silently dropping it changes the computed function |
| a non-constant weight | weight-stationary hardware has nothing to make resident |
| `uint8` operands | `cim.mvm` is signed; a `uint8` 200 read as int8 computes with −56 |
| non-zero matmul zero points | needs a per-output bias term the dialect cannot express |
| symbolic/dynamic dimensions | weights are materialized as dense literals |
| a bridge with a non-positive scale, or a missing/non-zero zero point | `cim.requantize` needs a positive scale and an explicit int8 zero point of 0 |
| a value read by more than one node along a chain | needs buffer-liveness reasoning this importer does not do; only a strictly linear chain is imported |
| a grouped/depthwise conv (`group != 1`) chained into further convs, or composed with pooling | accepted standalone or feeding a matmul chain (see above); those two loaders don't yet thread a grouped shape through their channel bookkeeping |
| a conv with non-zero `w_zero_point` | its correction term is per-row and activation-dependent, not a fixed per-channel bias |
| a conv with a non-scalar `x_zero_point`/`y_zero_point` | per-tensor activation quantization is the near-universal convention |
| `auto_pad` other than `NOTSET`, on conv or pool | needs shape-dependent padding math this front end does not replicate. `VALID` is refused too, because `onnx.reference`'s own formula for it is wrong for `N != 1` or `Cin != 1` — an oracle bug, sidestepped rather than reproduced |
| a chain of convs that is not strictly linear | branching, disconnected, or cyclic |
| a `MaxPool` with `strides == 1` | `onnx.reference` cannot evaluate an integer `MaxPool` at stride 1 at all — a gap in what can be *verified*, not what can be compiled |
| a `MaxPool` with `ceil_mode == 1` | this front end's output-size formula is floor-based |
| a `MaxPool` fed by a matmul layer, or before the chain's own first conv | pooling needs a real gather source on both sides — feeding a matmul is fine (max is scale-invariant), being fed by one is not |

## Two things worth knowing

**The activation is required.** The emitted module is `func.func @main()` with no
arguments, so it is one baked inference and the activation is a constant. There is
deliberately no default: `0 @ W == 0` for every `W`, so a zero default would make every
downstream numerical check pass against any weight matrix at all — including one that was
never transposed. Pass `--input` (`.npy` or `.json`), or `--input-random <seed>`, which is
recorded in the module's provenance header.

**The transpose.** ONNX stores the weight as `[K, N]`; `cim.mvm` indexes `W[n][k]`, so the
importer emits `Bᵀ`. This is the single most dangerous thing here: getting it backwards
produces structurally perfect IR and wrong numbers, and on a square weight it is completely
silent. Guarded three ways — rectangular shapes throughout, a direct assertion on the
emitted literal, and a mutation check proving that assertion has teeth.

## Rounding

`cim.requantize` rounds half-away-from-zero; ONNX's `QuantizeLinear` rounds half-to-even.
They differ only at an exact tie. At `scale=1.0` there is never a fractional part, so no
tie exists. At an **odd** integer scale a tie is mathematically impossible, so those chains
also match exactly. At an even scale a genuine tie can occur, and a test pins that
divergence explicitly rather than hiding it.

## Which passes to run downstream

`cim-detect`, `cim-partition` and `cim-placement` reproduce the model's arithmetic exactly.
The full eight-pass chain also runs `cim-legalize-precision`, which clamps every
accumulator to the target's `output_effective_bits`. That is a real modeled hardware
effect, not a bug — but it means the result stops matching an *unquantized* ONNX oracle as
soon as an accumulator leaves `[-128, 127]`.

## Analyzing a real model, without compiling it

`--emit-workload` answers a narrower question: how much weight-residency pressure does this
network put on a given chip? No activation needed, and the whole graph need not be
offloadable.

```sh
cim-import-onnx real_model.onnx --emit-workload -o workload.json
cim-bench analyze --target erbium-8t --workload-file workload.json --out results.json
```

This works because **weight-residency cost is a function of shape, not execution** — the
placement engine only ever needs each layer's `[K, N]`. So a `MaxPool`, `Concat`,
`Softmax`, or grouped convolution this front end cannot *compile* does not block *analysis*
of the layers around it.

Unlike every path above, this walk never refuses the whole model: each node becomes either
an offloadable layer or a skip with a stated reason, and the walk always completes.

**This is not end-to-end inference cost, and every output says so** — both the JSON and
`cim-bench analyze`'s own output state how many layers were analyzed, how many ops were
skipped, and why.

## Tests

```sh
pip install -r test/python/requirements-onnx.txt
```

Without the optional dependencies these skip; they never fail.

| Test | Covers |
|---|---|
| `test_onnx_emitter.py` | Emitted IR shape. Needs no ONNX and no build. |
| `test_onnx_frontend.py` | The end-to-end differential against `onnx.reference` and `onnxruntime`, including a batched case |
| `test_onnx_frontend_refusals.py` | One test per refusal above, plus a case proving it does not refuse everything |
| `test_onnx_frontend_chain.py` | Multi-layer matmul chains, the clamp, placement invariance, real calibrated scales, the known tie divergence |
| `test_onnx_frontend_conv.py` | Convolution: strided/padded, batched, per-channel scale, bias, asymmetric zero points, dilation, uint8 output, plus `onnx`-free unit tests of `im2col_nchw` |
| `test_onnx_frontend_conv_matmul_chain.py` | Conv feeding matmuls, and the bridge refusals |
| `test_onnx_frontend_conv_chain.py` | Conv-to-conv chains, including the channel-last weight flatten |
| `test_onnx_frontend_conv_chain_matmul_chain.py` | A conv stem feeding a fully-connected head |
| `test_onnx_frontend_conv_pool_chain.py` | `MaxPool`, against an independent hand-written NumPy oracle |
| `test_onnx_frontend_conv_pool_chain_matmul_chain.py` | Pooling composed with the conv->matmul bridge: a trailing pool alone, interior and trailing together, padded, batched, anti-vacuity, and the dispatch regression guard for the unpooled case |
| `test_analyze.py`, `test_workload_json_differential.py`, `test_cim_bench_analyze.py` | `--emit-workload` and the C++ JSON reader that consumes it |

Each differential also carries an **anti-vacuity check**: a deliberate perturbation in an
interior layer must actually be caught, so a test that passes for the wrong reason gets
found.
