# The ONNX front end

Reads an ONNX model and emits the MLIR the `cim` pipeline consumes. This
is the front door the rest of the project was written behind: everything
else here starts from MLIR that this repository wrote, either by hand in
the FileCheck tests or from a string builder in the differential tests.

```sh
PYTHONPATH=python python3 -m cim_frontend model.onnx --input act.npy \
  | cim-opt - --cim-detect \
      --cim-partition=target-yaml=targets/erbium-8t.yaml \
      --cim-placement=target-yaml=targets/erbium-8t.yaml \
  | cim-run --target-yaml=targets/erbium-8t.yaml --profile -
```

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

A graph consisting of exactly one `MatMulInteger` node and nothing else,
where:

- the weight operand (B) is a constant initializer,
- both operands are `int8` (not `uint8`),
- zero points are absent or zero,
- operands are rank 2 and the activation has a single row,
- every dimension is a known constant,
- the model targets opset 10 or later.

`MatMulInteger` is the accepted op because it *is* the v0.1 contract in
one node: int8 A, int8 B, int32 Y — "INT8 in, wider integer accumulator
out".

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
| more than one output row | v0.1 is a matrix-*vector* contract |
| symbolic/dynamic dimensions | weights are materialized as dense literals |

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

## Tests

- `test/python/test_onnx_emitter.py` — needs no ONNX and no build; pins
  the emitted IR shape against the pipeline's own module builder.
- `test/python/test_onnx_frontend.py` — the end-to-end differential
  against `onnx.reference` (the spec's own implementation) and
  `onnxruntime`.
- `test/python/test_onnx_frontend_refusals.py` — one test per refusal
  above, plus a case proving it does not refuse everything.
- `test/Transforms/onnx-imported-matmul.mlir` — importer output checked
  in, so the `mlir` CI job guards the shape without the ONNX packages.

Install the optional dependencies with:

```sh
pip install -r test/python/requirements-onnx.txt
```

Without them these tests skip; they never fail.
