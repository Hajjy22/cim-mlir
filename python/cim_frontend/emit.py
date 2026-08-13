"""MLIR text emission for the ONNX front end.

Deliberately free of any `onnx` dependency: this module turns plain numpy
arrays into MLIR text, so it can be imported -- and tested -- in a build
that has no model-reading libraries installed at all. `onnx_import.py` is
the only place that knows what a model file is.

THE IR SHAPE HERE IS NOT FREE TO VARY. It must stay byte-identical, for
the single-layer, single-ROW case, to `build_module()` in
test/python/test_numerical_differential.py, which is in turn the same
shape as `buildModule()` in test/mlir/pipeline_e2e_test.cpp. That shape is
the one the entire eight-pass pipeline is tested against, and two details
of it look redundant and are not:

  * The activation is staged through `memref.alloc` + `memref.copy` rather
    than used straight from its global. cim-detect counts constant
    operands and requires exactly ONE (lib/Transforms/CIMDetect.cpp,
    `isWeightOperand`): a matmul of two constants has nothing to make
    resident, so it is not a CIM candidate at all. Using the global
    directly makes the whole module silently fail to offload -- no error,
    just no cim.program anywhere.
  * Weights are output-major [N x K], not [K x N]. That is `cim.mvm`'s
    convention, and what `cimrt_mvm` in runtime/src/simulator/simulator.cpp
    actually indexes: out[r] = sum_c W[r*cols + c] * act[c].

test/python/test_onnx_frontend.py pins the first property by comparing
this module's output against `build_module` byte for byte.
"""

import numpy as np

# MLIR symbol names are [A-Za-z_$.][A-Za-z0-9_$.]* -- ONNX tensor names are
# not, and routinely contain '/', ':' and '-'.
_SAFE_FIRST = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$.")
_SAFE_REST = _SAFE_FIRST | set("0123456789")


def sanitize_symbol(name, taken):
    """An MLIR-legal symbol for `name`, unique against everything in `taken`.

    `taken` is mutated. Uniqueness is the point, not prettiness: two
    distinct ONNX tensors that sanitize to the same string would end up
    sharing one `memref.global`, so two layers would silently compute
    against the same weights. The IR would verify and the numbers would be
    wrong -- exactly the failure mode this project refuses -- so a
    collision gets a numeric suffix rather than a warning.
    """
    # Every position is sanitized against the same (wider) alphabet, and a
    # leading character that is legal-but-not-legal-first is PREFIXED
    # rather than replaced. Replacing it would silently discard
    # information -- '0abc' and '9abc' would both become '_abc' and then
    # need the collision suffix to tell them apart, which is a worse name
    # for no reason.
    base = "".join(ch if ch in _SAFE_REST else "_" for ch in name) or "_"
    if base[0] not in _SAFE_FIRST:
        base = "_" + base

    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _dense_2d(arr):
    """A 2-D dense<> literal body, without the enclosing `dense<...>`."""
    rows = ", ".join(
        "[" + ", ".join(str(int(v)) for v in row) + "]" for row in arr)
    return f"[{rows}]"


def emit_module(weight, activation, weight_sym="w", act_sym="a", header="",
                trailing_requantize=None, bias=None,
                per_channel_requantize=None, effective_bits=8):
    """A self-contained module computing `activation @ weight.T`.

    `weight` is [N, K] int8 -- ALREADY TRANSPOSED into cim.mvm's
    output-major convention. This function does not transpose: the ONNX
    -> output-major flip happens once, at a named boundary in
    onnx_import.py, so exactly one place in the front end can get it
    wrong and exactly one test has to guard it.

    `activation` is a length-K int8 vector (M == 1, the common case), or
    an [M, K] int8 array for a batched (M > 1) matmul -- cim-partition
    tiles a real M > 1 candidate the same way it tiles M == 1, generating
    a real scf.for over the rows itself (docs/roadmap.md's M4 entry), so
    this front end no longer has to refuse anything but [1, K]/[K].

    `trailing_requantize`, if given, is a `(scale, zero_point,
    effective_bits)` tuple: a `cim.requantize` is emitted after the matmul
    (and after `bias`, if also given) and its int8 result is what gets
    printed, instead of the raw int32 accumulator. This is for callers
    whose ONNX op ALWAYS quantizes its own output in one node --
    QLinearConv, unlike MatMulInteger, has no "raw int32" variant to
    import -- so unlike emit_chain_module's per-BRIDGE requantize (0 or 1
    per adjacent layer pair), this is a single, unconditional one attached
    to the only layer this function ever emits. Default `None` preserves
    this function's original raw-int32-output behavior exactly, including
    byte-identical text against test/python/test_numerical_differential.py's
    `build_module()`.

    `bias`, if given, is a length-N int32 array (one value per output
    channel), added to the RAW accumulator -- before any requantize --
    broadcast across every one of the M rows. Emitted as
    `cim.reduce_partial(matmul_output, bias_broadcast)`: reduce_partial's
    own verifier only constrains operand shape and element type, not
    producer, and cim-partition treats a matmul's `outs` buffer as an
    opaque, pre-allocated destination it fills in place (op->erase()
    after writing the real tiled result back into that SAME buffer) --
    so a hand-emitted reduce_partial reading that buffer composes with
    partition's own rewrite with no special-casing on either side.
    Verified by hand against a real cim-opt/cim-run round trip before this
    parameter was added: `a @ w.T + bias`, exactly, for an asymmetric
    (non-symmetric-per-row) bias so a row/column mix-up would be a loud,
    wrong number.

    `per_channel_requantize`, if given, is a list of N `(scale,
    zero_point)` pairs -- one per output channel -- mutually exclusive
    with `trailing_requantize`. A single `cim.requantize` call takes only
    one scalar `scale`/`zero_point` pair, so N of them are emitted
    instead, each against a `memref.subview` of one output column (a
    strided, not contiguous, view -- `cim.requantize`'s `AnyMemRef`
    argument accepts it, and the interpreter's gather/scatter already
    handle strided memrefs generically, the same machinery `cim-partition`
    itself relies on for ragged tiles), then copied into the matching
    column of the final result buffer. `effective_bits` (default 8) is
    shared across every channel, matching the target contract's own
    single per-target `output_effective_bits`. Verified by hand the same
    way as `bias`: two channels at deliberately different scales, so a
    channel mix-up computes a loud, wrong number rather than a close one.

    With the default symbol names, no header, and no other parameters
    beyond a 1-D (or [1, K]) activation, this returns text byte-identical
    to `build_module()` -- every parameter here is additive, not a
    rewrite of that base M == 1 shape.
    """
    weight = np.asarray(weight)
    activation = np.asarray(activation)
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [N, K], got shape {weight.shape}")
    if activation.ndim == 1:
        activation = activation.reshape(1, -1)
    elif activation.ndim != 2:
        raise ValueError(
            f"activation must be 1-D [K] or 2-D [M, K], got shape "
            f"{activation.shape}")

    n, k = weight.shape
    m = activation.shape[0]
    if activation.shape[1] != k:
        raise ValueError(
            f"activation's K of {activation.shape[1]} does not match the "
            f"weight's K of {k}")
    if trailing_requantize is not None and per_channel_requantize is not None:
        raise ValueError(
            "trailing_requantize and per_channel_requantize are mutually "
            "exclusive -- pass one uniform scale or N per-channel ones, "
            "not both")
    if per_channel_requantize is not None and len(per_channel_requantize) != n:
        raise ValueError(
            f"per_channel_requantize must have one (scale, zero_point) "
            f"pair per output channel (N = {n}), got "
            f"{len(per_channel_requantize)}")

    rows = _dense_2d(weight)
    acts = _dense_2d(activation)

    # (name, type) for every memref this function allocates, in allocation
    # order, so every one gets exactly one memref.dealloc -- an ASan build
    # (LeakSanitizer is on by default alongside address) would flag a
    # missed one as a real leak, not just untidy IR.
    allocs = [("%out", f"memref<{m}x{n}xi32>")]
    lines = [
        f"  %out = memref.alloc() : memref<{m}x{n}xi32>",
        f"  linalg.matmul_transpose_b ins(%a, %w : memref<{m}x{k}xi8>, "
        f"memref<{n}x{k}xi8>)",
        f"    outs(%out : memref<{m}x{n}xi32>)",
    ]
    acc_val = "%out"
    bias_global = ""

    if bias is not None:
        bias = np.asarray(bias)
        if bias.shape != (n,):
            raise ValueError(
                f"bias must be a length-N ({n}) int32 array, got shape "
                f"{bias.shape}")
        broadcast = np.tile(bias.reshape(1, n), (m, 1))
        bias_global = (
            f'memref.global "private" constant @{weight_sym}_bias : '
            f"memref<{m}x{n}xi32> = dense<{_dense_2d(broadcast)}>\n")
        i32mn = f"memref<{m}x{n}xi32>"
        lines += [
            f"  %biasInit = memref.get_global @{weight_sym}_bias : {i32mn}",
            f"  %bias = memref.alloc() : {i32mn}",
            f"  memref.copy %biasInit, %bias : {i32mn} to {i32mn}",
            f"  %biased = cim.reduce_partial %out, %bias : ({i32mn}, "
            f"{i32mn}) -> {i32mn}",
        ]
        # cim.reduce_partial always produces a fresh allocation (see
        # Interpreter.cpp's runReducePartial: makeAllocation, never an
        # in-place write into an operand), so %biased needs its own
        # dealloc distinct from %out's.
        allocs += [("%bias", i32mn), ("%biased", i32mn)]
        acc_val = "%biased"

    if per_channel_requantize is not None:
        i8mn = f"memref<{m}x{n}xi8>"
        lines.append(f"  %result = memref.alloc() : {i8mn}")
        allocs.append(("%result", i8mn))
        for i, (scale, zero_point) in enumerate(per_channel_requantize):
            if not (float(scale) > 0.0):
                raise ValueError(
                    f"per_channel_requantize[{i}]'s scale must be "
                    f"positive, got {scale}")
            src_ty = f"memref<{m}x1xi32, strided<[{n}, 1], offset: {i}>>"
            dst_ty = f"memref<{m}x1xi8, strided<[{n}, 1], offset: {i}>>"
            col_ty = f"memref<{m}x1xi8>"
            lines += [
                f"  %col{i} = memref.subview {acc_val}[0, {i}] [{m}, 1] "
                f"[1, 1] : memref<{m}x{n}xi32> to {src_ty}",
                f"  %q{i} = cim.requantize %col{i} {{scale = "
                f"{float(scale)!r} : f32, zero_point = {int(zero_point)} : "
                f"i32, effective_bits = {int(effective_bits)} : i32}}",
                f"    : {src_ty} -> {col_ty}",
                f"  %rcol{i} = memref.subview %result[0, {i}] [{m}, 1] "
                f"[1, 1] : {i8mn} to {dst_ty}",
                f"  memref.copy %q{i}, %rcol{i} : {col_ty} to {dst_ty}",
            ]
            # %q{i} is cim.requantize's own fresh allocation (same
            # reasoning as %biased above), and outlives its one use
            # (the copy into %result) until this function's own cleanup.
            allocs.append((f"%q{i}", col_ty))
        result_val, result_elem = "%result", "i8"
    elif trailing_requantize is not None:
        scale, zero_point, eff_bits = trailing_requantize
        if not (float(scale) > 0.0):
            raise ValueError(
                f"trailing_requantize's scale must be positive, got {scale}")
        i8mn = f"memref<{m}x{n}xi8>"
        lines += [
            f"  %q = cim.requantize {acc_val} {{scale = {float(scale)!r} : "
            f"f32, zero_point = {int(zero_point)} : i32, effective_bits = "
            f"{int(eff_bits)} : i32}}",
            f"    : memref<{m}x{n}xi32> -> {i8mn}",
        ]
        allocs.append(("%q", i8mn))
        result_val, result_elem = "%q", "i8"
    else:
        result_val, result_elem = acc_val, "i32"

    quantized = trailing_requantize is not None or per_channel_requantize is not None
    print_call = "cim_print_i8" if quantized else "cim_print_i32"
    result_type = f"memref<{m}x{n}x{result_elem}>"
    unranked_type = f"memref<*x{result_elem}>"
    body = "\n".join(lines)
    dealloc_lines = "\n".join(
        f"  memref.dealloc {name} : {ty}" for name, ty in allocs)
    return f"""\
{header}memref.global "private" constant @{weight_sym} : memref<{n}x{k}xi8> = dense<{rows}>
memref.global "private" constant @{act_sym} : memref<{m}x{k}xi8> = dense<{acts}>
{bias_global}func.func private @{print_call}({unranked_type})
func.func @main() {{
  %w = memref.get_global @{weight_sym} : memref<{n}x{k}xi8>
  %aInit = memref.get_global @{act_sym} : memref<{m}x{k}xi8>
  %a = memref.alloc() : memref<{m}x{k}xi8>
  memref.copy %aInit, %a : memref<{m}x{k}xi8> to memref<{m}x{k}xi8>
{body}
  %u = memref.cast {result_val} : {result_type} to {unranked_type}
  func.call @{print_call}(%u) : ({unranked_type}) -> ()
  memref.dealloc %a : memref<{m}x{k}xi8>
{dealloc_lines}
  return
}}
"""


def emit_chain_module(weights, activation, weight_syms=None, act_sym="a",
                      header="", scales=None):
    """A chain of matmuls, each layer's accumulator bridged into the next
    via a real `cim.requantize` -- int8 in, int32 accumulator out, narrowed
    back to int8 before it becomes the next layer's activation.

    `scales`, if given, is a list of `len(weights) - 1` positive floats,
    one per bridge, in the same order as `weights` -- a real, calibrated
    per-layer requantization scale, exactly the shape a real post-training-
    quantized multi-layer INT8 model uses. Defaults to 1.0 for every
    bridge, which is why passing nothing is safe and unchanged from
    before this parameter existed.

    Why `scale=1.0` (the default) is safe to compare against an
    UNQUANTIZED ONNX oracle, despite requantize normally being the thing
    that introduces modeled precision loss (see onnx_import.py's module
    docstring and test_onnx_frontend.py's header on why the differential
    otherwise avoids cim-legalize-precision entirely): with `scale=1.0`
    there is nothing to round -- the accumulator is already an integer, so
    round(v / 1.0) == v exactly, and round-half-to-even (ONNX
    QuantizeLinear) cannot diverge from round-half-away-from-zero
    (cim.requantize) when there is never a tie to break. Both sides then
    saturate to the same [-128, 127]. So `cim.requantize(scale=1.0,
    zero_point=0, effective_bits=8)` computes bit-for-bit the same
    function as ONNX's `Cast(to=float32) -> QuantizeLinear(scale=1.0,
    zero_point=0)` pair.

    A REAL (non-1.0) scale does NOT get the same guarantee: `v / scale`
    can land exactly on a tie (a value ending in .5), and at that exact
    point cim.requantize's round-half-away-from-zero and QuantizeLinear's
    round-half-to-even genuinely, correctly disagree by 1 -- a real
    hardware-fidelity fact (cim.requantize's rounding mode models a real
    digital requantizer/ADC readout path; it is not a testing
    convenience, and changing it to match ONNX would model the wrong
    thing), not a bug. onnx_import.py's `_validate_bridge` accepts any
    positive scale for exactly this reason: the earlier "scale must be
    1.0" refusal existed only to keep the differential trivially exact,
    not because a real scale was unsupported -- cim.requantize's own
    parameters were already scale/zero_point/effective_bits-generic (see
    test/mlir/legalize_precision_e2e_test.cpp). test_onnx_frontend_chain.py
    checks a real-scale chain against onnx.reference's OWN full
    (quantized) evaluation instead of an unquantized one, and separately,
    explicitly documents the exact-tie divergence rather than letting it
    surface as a mysterious near-miss.

    This was verified by hand against a real cim-opt/cim-run round trip
    (--cim-detect --cim-partition --cim-placement, no
    cim-legalize-precision) before this function was written: a 2-layer
    chain's compiled output matched `np.clip(layer0_output, -128,
    127).astype(np.int8)` fed into layer 1, exactly, at scale=1.0.

    `weights` is a list of already-transposed [N_i, K_i] int8 arrays, with
    K_i == N_(i-1) for i > 0. `activation` is a length-K_0 int8 vector
    (M == 1), or an [M, K_0] int8 array for a batched (M > 1) chain -- see
    `emit_module`'s own note on why this is additive, not a rewrite, for
    M == 1. `weight_syms`, if given, must have one entry per layer.

    M threads through every layer unchanged: cim.requantize does not
    change shape (its own verifier enforces exactly that), so a batched
    layer 0 activation makes every later layer's matmul and requantize
    genuinely [M, ...] too, with no per-layer M bookkeeping needed here.
    """
    weights = [np.asarray(w) for w in weights]
    activation = np.asarray(activation)
    if not weights:
        raise ValueError("emit_chain_module needs at least one layer")
    if weight_syms is None:
        weight_syms = [f"w{i}" for i in range(len(weights))]
    if len(weight_syms) != len(weights):
        raise ValueError("weight_syms must have one entry per layer")
    if scales is None:
        scales = [1.0] * (len(weights) - 1)
    if len(scales) != len(weights) - 1:
        raise ValueError(
            f"scales must have one entry per bridge (len(weights) - 1 = "
            f"{len(weights) - 1}), got {len(scales)}")
    for i, s in enumerate(scales):
        if not (s > 0.0):
            raise ValueError(f"bridge {i}'s scale must be positive, got {s}")

    for i, w in enumerate(weights):
        if w.ndim != 2:
            raise ValueError(f"layer {i}'s weight must be 2-D [N, K], got "
                             f"shape {w.shape}")
    for i in range(1, len(weights)):
        if weights[i].shape[1] != weights[i - 1].shape[0]:
            raise ValueError(
                f"layer {i}'s K ({weights[i].shape[1]}) does not match "
                f"layer {i - 1}'s N ({weights[i - 1].shape[0]})")
    if activation.ndim == 1:
        activation = activation.reshape(1, -1)
    elif activation.ndim != 2:
        raise ValueError(
            f"activation must be 1-D [K] or 2-D [M, K], got shape "
            f"{activation.shape}")
    if activation.shape[1] != weights[0].shape[1]:
        raise ValueError(
            f"activation's K of {activation.shape[1]} does not match layer "
            f"0's K of {weights[0].shape[1]}")
    m = activation.shape[0]

    lines = [header.rstrip("\n")] if header else []
    for sym, w in zip(weight_syms, weights):
        n, k = w.shape
        lines.append(
            f'memref.global "private" constant @{sym} : memref<{n}x{k}xi8> '
            f"= dense<{_dense_2d(w)}>")
    k0 = weights[0].shape[1]
    lines.append(
        f'memref.global "private" constant @{act_sym} : memref<{m}x{k0}xi8> '
        f"= dense<{_dense_2d(activation)}>")
    lines.append("func.func private @cim_print_i32(memref<*xi32>)")
    lines.append("func.func @main() {")

    for sym, w in zip(weight_syms, weights):
        n, k = w.shape
        lines.append(f"  %{sym} = memref.get_global @{sym} : memref<{n}x{k}xi8>")
    lines.append(f"  %aInit = memref.get_global @{act_sym} : memref<{m}x{k0}xi8>")
    lines.append(f"  %a0 = memref.alloc() : memref<{m}x{k0}xi8>")
    lines.append(
        f"  memref.copy %aInit, %a0 : memref<{m}x{k0}xi8> to memref<{m}x{k0}xi8>")

    act_val = "%a0"
    for i, (sym, w) in enumerate(zip(weight_syms, weights)):
        n, k = w.shape
        out_val = f"%out{i}"
        lines.append(f"  {out_val} = memref.alloc() : memref<{m}x{n}xi32>")
        lines.append(
            f"  linalg.matmul_transpose_b ins({act_val}, %{sym} : "
            f"memref<{m}x{k}xi8>, memref<{n}x{k}xi8>)")
        lines.append(f"    outs({out_val} : memref<{m}x{n}xi32>)")
        if i < len(weights) - 1:
            act_val = f"%act{i + 1}"
            lines.append(
                f"  {act_val} = cim.requantize {out_val} {{scale = "
                f"{float(scales[i])!r} : f32, "
                f"zero_point = 0 : i32, effective_bits = 8 : i32}}")
            lines.append(f"    : memref<{m}x{n}xi32> -> memref<{m}x{n}xi8>")

    last_n = weights[-1].shape[0]
    lines.append(
        f"  %u = memref.cast %out{len(weights) - 1} : memref<{m}x{last_n}xi32> "
        f"to memref<*xi32>")
    lines.append("  func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()")
    lines.append(f"  memref.dealloc %a0 : memref<{m}x{k0}xi8>")
    lines.append(
        f"  memref.dealloc %out{len(weights) - 1} : memref<{m}x{last_n}xi32>")
    lines.append("  return")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
