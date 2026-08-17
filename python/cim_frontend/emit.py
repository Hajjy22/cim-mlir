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

from .im2col import output_size

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


def _emit_bias_and_requantize(acc_val, m, n, weight_sym, bias=None,
                              per_channel_requantize=None,
                              trailing_requantize=None, effective_bits=8):
    """Shared tail: an optional bias-add, then an optional requantize,
    applied to an `[m, n]` int32 accumulator named `acc_val`.

    Factored out of `emit_module`'s own original inline logic so
    `emit_conv_chain_module` can apply the EXACT same tail to a
    conv-terminated chain's FINAL layer. That layer's real ONNX identity
    is a `QLinearConv`, whose `"Y"` output is defined by the op itself as
    the fully quantized result (bias, then per-channel or scalar
    requantize, already applied) -- NOT a raw accumulator, unlike
    `MatMulInteger`'s own `"Y"`, which the matmul-chain path can print
    as-is. Printing the raw accumulator for a conv-terminated chain would
    silently misrepresent what the graph's own declared output actually
    is -- exactly the "structurally valid IR, confidently wrong number"
    failure class this project exists to prevent -- so this helper exists
    to make both call sites share one, single-source-of-truth
    implementation rather than risk the two drifting apart.

    Returns `(lines, allocs, bias_global, result_val, result_elem)`:
      * `lines` -- new MLIR body text, to append right after the matmul
        (or last layer's matmul) that produced `acc_val`.
      * `allocs` -- `(name, type)` pairs needing their own
        `memref.dealloc`, in allocation order (same ASan-driven
        bookkeeping `emit_module` already does for its own allocations).
      * `bias_global` -- a `memref.global` declaration string (empty if
        `bias` is `None`) that MUST be placed at MODULE scope, before
        `func.func @main()` opens -- unlike `lines`/`allocs`, the caller
        is responsible for where this one goes.
      * `result_val`, `result_elem` -- the SSA name and element type
        (`"i8"` if either requantize path ran, else `"i32"`) of the final
        value to print.
    """
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

    lines = []
    allocs = []
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
            f"  %biased = cim.reduce_partial {acc_val}, %bias : ({i32mn}, "
            f"{i32mn}) -> {i32mn}",
        ]
        # cim.reduce_partial always produces a fresh allocation (see
        # Interpreter.cpp's runReducePartial: makeAllocation, never an
        # in-place write into an operand), so %biased needs its own
        # dealloc distinct from acc_val's.
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
            # (the copy into %result) until the caller's own cleanup.
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

    return lines, allocs, bias_global, result_val, result_elem


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
    tail_lines, tail_allocs, bias_global, result_val, result_elem = (
        _emit_bias_and_requantize(
            "%out", m, n, weight_sym, bias=bias,
            per_channel_requantize=per_channel_requantize,
            trailing_requantize=trailing_requantize,
            effective_bits=effective_bits))
    lines += tail_lines
    allocs += tail_allocs

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


def emit_conv_chain_module(weights2d, conv_params, patches0, out_shape0,
                           weight_syms=None, act_sym="a", header="",
                           scales=None, n_conv_layers=None, last_bias=None,
                           last_trailing_requantize=None,
                           last_per_channel_requantize=None,
                           effective_bits=8, pool_params=None):
    """Two or more chained QLinearConv layers -- a REAL MLIR-level im2col
    for every layer after the first, docs/roadmap.md's "chain convolution
    layers" plan, part 2.

    Layer 0's activation is still a Python-computed constant (the literal
    graph input), so its own im2col already ran in Python exactly as it
    does for a standalone conv -- `patches0`/`out_shape0` are exactly
    load_qlinear_conv's own `im2col_nchw` output for that layer. Layer
    i >= 1's activation is layer i-1's own bridged output, an MLIR SSA
    value this function has never seen as a concrete array -- gathering
    its im2col patches has to happen IN THE EMITTED IR, which is the new
    capability this function adds.

    THE DESIGN, AND WHY IT LOOKS LIKE THIS
    ==========================================
    Confirmed by hand against the real cim-opt/cim-run binaries before
    this function was written: the interpreter's execute() is a closed
    TypeSwitch that cannot evaluate `arith.addi`/`muli` or `scf.if` at
    all. That rules out the two "obvious" im2col designs -- an scf.for
    computing tap indices via arithmetic, and a fully-unrolled emission
    over every output position (IR size scales with OutH*OutW, easily
    thousands) -- and is why this one looks the way it does: unroll over
    Kh*Kw kernel TAPS (small -- 9 to 49 for a realistic kernel) instead,
    each one a single FULLY STATIC, non-unit-stride memref.subview +
    memref.copy covering every output position for that tap in one shot.
    memref.subview already supports a static stride generically (see
    Interpreter.cpp's own runSubView); no loop, no arithmetic, needed.

    Per bridge (layer i -> layer i+1, i from 0 to len(weights2d)-2):
      1. `cim.requantize` layer i's raw int32 accumulator into int8 --
         exactly emit_chain_module's own bridge, reused unchanged.
      2. `memref.expand_shape` that bridged [M_i, Cout_i] activation into
         a 4-D [N, H_i, W_i, Cout_i] view -- M_i's own (n, oh, ow)
         row-major order (im2col_nchw's own convention, and what every
         earlier bridge/matmul in this chain already commits to) IS this
         reshape, not a guess at one.
      3. A padded 4-D buffer (`memref.alloc` + `linalg.fill` zero +
         `memref.copy` the interior in) -- the direct MLIR analogue of
         im2col.py's own `_pad_nchw`: pad AFTER any zero-point shift
         already happened (there is none here -- intermediate zero_point
         is required to be 0, see load_conv_chain's own module header),
         so zero is always the logically correct pad value. Always
         allocated, even when this layer's own padding is zero, for
         emission simplicity -- a redundant alloc+fill+copy costs
         nothing real at this scale.
      4. Kh*Kw taps, each one static-subview + copy into a fresh
         contiguous slab + `memref.collapse_shape` back to 2-D + a copy
         into the matching column-block of layer i+1's own patches
         buffer -- exactly the composition hand-verified against a real
         cim-opt/cim-run round trip (and an independent, non-im2col
         manual convolution, for a real 2-layer chain with batching,
         padding, stride, AND dilation on the interior layer all
         together) before this function was written.

    `weights2d[i]` is layer i's weight, ALREADY reshaped to [Cout_i, K_i]:
    Cin-major ([Cout, Cin*Kh*Kw], load_qlinear_conv's own convention, no
    transpose) for layer 0; CHANNEL-LAST ([Cout, Kh*Kw*Cin]) for layer
    i >= 1 -- THE dangerous line in this feature, the same shape of bug
    as onnx_import.py's own "THE TRANSPOSE" note: layer i>=1's activation
    is NHWC (channel-last) by construction of how this function's own
    gather lays out patches columns (tap-major, channel-minor within each
    tap), so its weight must flatten to match, the OPPOSITE order from
    layer 0's NCHW-sourced, Cin-major flatten. Getting this backwards
    produces IR that verifies, compiles, and runs, and computes the wrong
    answer -- see its own named test and mutation test.

    `conv_params[i]` (i in 1..len(weights2d)-1) is layer i's own
    `(kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left,
    pad_right, dilation_h, dilation_w)` -- exactly im2col_nchw's own
    trailing arguments -- describing how to gather layer i-1's bridged
    output into layer i's matmul input. `conv_params[0]` is unused.

    `scales` has `len(weights2d) - 1` entries, bridge i's real positive
    cim.requantize scale -- same convention, same "why any positive scale
    is safe, and what a real (non-1.0) one changes about which oracle to
    compare against" as emit_chain_module's own `scales` parameter.

    `n_conv_layers`, if given, is the number of LEADING layers that are
    real convolutions needing the gather machinery above (default: every
    layer -- this function's original, PR C scope, unchanged). Layers from
    `n_conv_layers` onward are plain `MatMulInteger` layers: no
    expand_shape/pad/tap-gather at all, just `emit_chain_module`'s own
    bridge-then-matmul step, since a matmul layer's activation is already
    the flat `[M, K]` shape a matmul needs -- unlike a conv layer, whose
    activation needs reinterpreting as a 4-D spatial view first. `M`
    threads through every matmul-only layer unchanged, exactly
    `emit_chain_module`'s own note. `conv_params[i]` for `i >=
    n_conv_layers` is unused (the matmul portion needs no geometry), the
    same "entry present but unused" convention `conv_params[0]` already
    has. This is the "chain convolution layers" plan's PR D: a real
    conv-stem-feeding-an-FC-head chain -- the realistic full CNN shape --
    built by composing PR C's own conv-chain gather with PR B's own
    conv-to-matmul bridge, both reused verbatim rather than reimplemented.

    `last_bias`, `last_trailing_requantize`, and `last_per_channel_
    requantize` apply ONLY when the chain's own FINAL layer is a
    convolution (`n_conv_layers == len(weights2d)`), with exactly
    `emit_module`'s own meaning for each (see its docstring) -- via the
    shared `_emit_bias_and_requantize` helper, so the two stay a single
    implementation. This matters because a conv-terminated chain's real
    ONNX identity is a `QLinearConv`, and unlike `MatMulInteger` (whose
    "Y" IS the raw accumulator, which is what every matmul-terminated
    chain prints, with these three parameters forbidden -- see the
    ValueError below), QLinearConv's own "Y" is DEFINED by the op itself
    as the fully quantized result -- bias added, then a scalar or
    per-channel requantize applied -- so a chain that truly ends in a conv
    needs this to faithfully reprint what the graph itself declares as
    its output. Default `None` for all three preserves this function's
    original raw-int32-final-accumulator behavior exactly (for either
    kind of final layer).

    `pool_params`, if given, is a `{bridge_index: (kh, kw, stride_h,
    stride_w, pad_top, pad_bottom, pad_left, pad_right, dilation_h,
    dilation_w)}` dict -- the SAME 10-tuple shape as one `conv_params[i]`
    entry, reusing it rather than inventing a second shape, since a
    MaxPool window's own geometry is validated the same way a
    convolution's is (see im2col.py's shared `output_size`). `bridge_index`
    i (0 <= i < n_conv_layers - 1, i.e. strictly inside the convolutional
    portion of the chain -- pooling into or out of a plain matmul layer is
    not supported here) means "insert a MaxPool between layer i's own
    bridge and layer i+1's own gather". A pool at bridge i is emitted
    right after that bridge's `cim.requantize` (which still runs
    unconditionally -- MaxPool carries no quantization of its own, so it
    consumes that requantize's own int8 output directly and needs no
    `scales[i]` entry of its own beyond it) and BEFORE layer i+1 does its
    own gather, as:
      1. `memref.expand_shape` the bridged `[M_i, Cout_i]` activation into
         4-D `[N, H_i, W_i, Cout_i]` -- identical to a conv layer's own
         first gather step.
      2. A padded 4-D buffer filled with **-128** (`INT8_MIN`), not 0 --
         the correct MaxPool pad value (see im2col.py's `_pad_nchw`
         docstring for the three-implementation confirmation), so a
         padded position never wins a max against a real one.
      3. `Kh*Kw` static-stride `memref.subview` taps of that padded
         buffer, fed DIRECTLY as `cim.reduce_max`'s variadic operands --
         no per-tap slab copy into a patches buffer the way a conv
         layer's gather needs: `cim.reduce_max`'s own verifier constrains
         only shape and element type, and the interpreter's gather walks
         arbitrary strides already, so the taps compose as-is (confirmed
         against a real cim-opt/cim-run round trip before this parameter
         existed -- see test/Run/reduce-max.mlir). One `cim.reduce_max`
         call folds all `Kh*Kw` taps into a single fresh 4-D allocation.
      4. `memref.collapse_shape` that 4-D result back to 2-D `[pool_M,
         Cout_i]` -- the shape layer i+1's own subsequent processing
         (gather or, if it were a matmul, a flat activation) expects.
    `M`/`H`/`W` after a pool are the pool's own output size, threaded into
    every later computation the normal way -- the chain's `layer_shapes`
    precompute and the main emission loop both apply a bridge's pool (if
    any) before computing the NEXT layer's own geometry from it. Default
    `None` (no pooling anywhere) leaves every layer's own bridge exactly
    as before this parameter existed.

    Deallocation follows emit_chain_module's own established convention
    (only `%a0` and the final PRINTED buffer get an explicit
    `memref.dealloc` -- `%out{n-1}` itself if no `last_*` tail ran, or
    whichever buffer the tail produced otherwise), not emit_module's
    stricter per-buffer one: this interpreter's `memref.dealloc` is a
    no-op (Interpreter.cpp's own TypeSwitch), and every real allocation is
    `std::shared_ptr`-owned, so it is freed exactly once, automatically,
    when the interpreter's own `memrefs` map is destroyed at the end of
    the run -- an "extra" live map entry until then is not a real leak,
    confirmed clean under the same MLIR+ASan+UBSan configuration CI's
    `mlir-asan` job uses.
    """
    weights2d = [np.asarray(w) for w in weights2d]
    patches0 = np.asarray(patches0)
    n_layers = len(weights2d)
    if n_layers < 2:
        raise ValueError("emit_conv_chain_module needs at least two conv layers")
    if len(conv_params) != n_layers:
        raise ValueError(
            f"conv_params must have one entry per layer (n_layers = "
            f"{n_layers}, entry 0 unused), got {len(conv_params)}")
    if n_conv_layers is None:
        n_conv_layers = n_layers
    if not (1 <= n_conv_layers <= n_layers):
        raise ValueError(
            f"n_conv_layers must be between 1 and {n_layers} (the total "
            f"layer count -- layer 0 is always a convolution, since its "
            f"activation is the Python-computed patches0, not a bare "
            f"matmul activation), got {n_conv_layers}")
    ends_in_conv = n_conv_layers == n_layers
    if not ends_in_conv and (last_bias is not None
                            or last_trailing_requantize is not None
                            or last_per_channel_requantize is not None):
        raise ValueError(
            "last_bias/last_trailing_requantize/last_per_channel_requantize "
            "only apply when the chain's own final layer is a convolution "
            "(n_conv_layers == len(weights2d)) -- a chain ending in a "
            "plain MatMulInteger layer prints that layer's own raw int32 "
            "accumulator directly, exactly like emit_chain_module, since "
            "MatMulInteger's own \"Y\" IS the raw accumulator by the op's "
            "own definition, needing no further tail.")
    if pool_params is None:
        pool_params = {}
    for i in pool_params:
        if not (0 <= i < n_conv_layers - 1):
            raise ValueError(
                f"pool_params has an entry for bridge {i}, but a pool is "
                f"only supported strictly inside the convolutional portion "
                f"of the chain (0 <= bridge < n_conv_layers - 1 = "
                f"{n_conv_layers - 1}) -- layer {i + 1} must itself be a "
                f"convolution for its own gather to consume the pooled "
                f"activation. Nothing emitted.")
    if weight_syms is None:
        weight_syms = [f"w{i}" for i in range(n_layers)]
    if len(weight_syms) != n_layers:
        raise ValueError("weight_syms must have one entry per layer")
    if scales is None:
        scales = [1.0] * (n_layers - 1)
    if len(scales) != n_layers - 1:
        raise ValueError(
            f"scales must have one entry per bridge (n_layers - 1 = "
            f"{n_layers - 1}), got {len(scales)}")
    for i, s in enumerate(scales):
        if not (s > 0.0):
            raise ValueError(f"bridge {i}'s scale must be positive, got {s}")
    for i, w in enumerate(weights2d):
        if w.ndim != 2:
            raise ValueError(f"layer {i}'s weight must be 2-D [Cout, K], "
                             f"got shape {w.shape}")

    n_batch, out_h, out_w = out_shape0
    m0, k0 = patches0.shape
    if m0 != n_batch * out_h * out_w:
        raise ValueError(
            f"patches0 has {m0} rows but out_shape0 implies "
            f"{n_batch * out_h * out_w} (N*OutH*OutW). Nothing emitted.")
    if weights2d[0].shape[1] != k0:
        raise ValueError(
            f"layer 0's weight K ({weights2d[0].shape[1]}) does not match "
            f"patches0's K ({k0})")
    for i in range(1, n_layers):
        if i < n_conv_layers:
            if weights2d[i].shape[1] != weights2d[i - 1].shape[0] * conv_params[i][0] * conv_params[i][1]:
                raise ValueError(
                    f"layer {i}'s weight K ({weights2d[i].shape[1]}) does "
                    f"not match Kh*Kw*Cin (layer {i - 1}'s "
                    f"Cout={weights2d[i - 1].shape[0]}, layer {i}'s "
                    f"Kh={conv_params[i][0]}, Kw={conv_params[i][1]})")
        else:
            # A plain matmul layer: K == the previous layer's own N/Cout
            # directly, no Kh*Kw spatial expansion -- load_matmul_chain's
            # own convention.
            if weights2d[i].shape[1] != weights2d[i - 1].shape[0]:
                raise ValueError(
                    f"layer {i}'s weight K ({weights2d[i].shape[1]}) does "
                    f"not match layer {i - 1}'s own N/Cout "
                    f"({weights2d[i - 1].shape[0]})")

    # Every layer's own (M, OutH, OutW), computed once, purely from static
    # shapes/geometry -- BEFORE any MLIR text is emitted. This is needed
    # because the final layer's own bias global (if `last_bias` is given)
    # must be declared at MODULE scope, before `func.func @main()` opens,
    # but its shape depends on the final layer's own M, which a
    # single-emission-pass design would only know after the whole
    # per-layer loop below had already run. Kept in exactly one place so
    # the loop's own per-layer gather geometry (which still needs each
    # layer's OutH/OutW to size its subviews) reads from here rather than
    # recomputing the same formula a second time and risking the two
    # silently drifting apart.
    # Only the CONV portion (indices 1..n_conv_layers-1) has real spatial
    # geometry to precompute -- a plain matmul layer (index >=
    # n_conv_layers) keeps M unchanged and has no OutH/OutW at all, so it
    # needs no entry here (nothing downstream reads past n_conv_layers-1:
    # see below).
    layer_shapes = [(m0, out_h, out_w)]
    ph, pw = out_h, out_w
    for i in range(1, n_conv_layers):
        # A pool on bridge (i - 1) -- between layer i-1's own bridge and
        # layer i's own gather -- changes (ph, pw) BEFORE layer i's own
        # geometry is applied to them; see this function's own
        # `pool_params` docstring.
        if (i - 1) in pool_params:
            (pkh, pkw, p_stride_h, p_stride_w, p_pad_top, p_pad_bottom,
             p_pad_left, p_pad_right, p_dilation_h, p_dilation_w) = (
                pool_params[i - 1])
            pool_out_h = output_size(ph, pkh, p_pad_top, p_pad_bottom,
                                     p_stride_h, p_dilation_h)
            pool_out_w = output_size(pw, pkw, p_pad_left, p_pad_right,
                                     p_stride_w, p_dilation_w)
            if pool_out_h <= 0 or pool_out_w <= 0:
                raise ValueError(
                    f"bridge {i - 1}'s pool: non-positive output size "
                    f"({pool_out_h}, {pool_out_w}) for input ({ph}, {pw}), "
                    f"kernel ({pkh}, {pkw}), stride ({p_stride_h}, "
                    f"{p_stride_w}), pad ({p_pad_top}, {p_pad_bottom}, "
                    f"{p_pad_left}, {p_pad_right})")
            ph, pw = pool_out_h, pool_out_w

        (kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left,
         pad_right, dilation_h, dilation_w) = conv_params[i]
        next_out_h = output_size(ph, kh, pad_top, pad_bottom, stride_h,
                                 dilation_h)
        next_out_w = output_size(pw, kw, pad_left, pad_right, stride_w,
                                 dilation_w)
        if next_out_h <= 0 or next_out_w <= 0:
            raise ValueError(
                f"layer {i}: non-positive output size "
                f"({next_out_h}, {next_out_w}) for input ({ph}, {pw}), "
                f"kernel ({kh}, {kw}), stride ({stride_h}, {stride_w}), "
                f"pad ({pad_top}, {pad_bottom}, {pad_left}, {pad_right}), "
                f"dilation ({dilation_h}, {dilation_w})")
        layer_shapes.append((n_batch * next_out_h * next_out_w,
                             next_out_h, next_out_w))
        ph, pw = next_out_h, next_out_w

    if ends_in_conv:
        last_m = layer_shapes[-1][0]
        last_cout = weights2d[-1].shape[0]
        # _tail_allocs (the per-buffer dealloc list _emit_bias_and_requantize
        # gives emit_module) is deliberately unused here: this function
        # keeps its OWN leaky-by-convention discipline (see the module
        # docstring above), not emit_module's stricter one, so only the
        # final PRINTED buffer (`result_val`, below) gets an explicit
        # dealloc.
        tail_lines, _tail_allocs, bias_global, result_val, result_elem = (
            _emit_bias_and_requantize(
                f"%out{n_layers - 1}", last_m, last_cout, weight_syms[-1],
                bias=last_bias,
                per_channel_requantize=last_per_channel_requantize,
                trailing_requantize=last_trailing_requantize,
                effective_bits=effective_bits))
        quantized = (last_trailing_requantize is not None
                    or last_per_channel_requantize is not None)
        print_call = "cim_print_i8" if quantized else "cim_print_i32"
        result_type = f"memref<{last_m}x{last_cout}x{result_elem}>"
    else:
        # A matmul-terminated chain prints its own last layer's raw int32
        # accumulator directly -- exactly emit_chain_module's own
        # convention -- so there is no tail to precompute, and result_val/
        # result_type are only known once the main loop below has actually
        # reached the final layer (its own M may have changed during the
        # conv portion).
        bias_global = ""
        print_call = "cim_print_i32"
        result_elem = "i32"
        tail_lines = None
        result_val = None
        result_type = None
    unranked_type = f"memref<*x{result_elem}>"

    lines = [header.rstrip("\n")] if header else []
    for sym, w in zip(weight_syms, weights2d):
        cout, k = w.shape
        lines.append(
            f'memref.global "private" constant @{sym} : memref<{cout}x{k}xi8> '
            f"= dense<{_dense_2d(w)}>")
    lines.append(
        f'memref.global "private" constant @{act_sym} : memref<{m0}x{k0}xi8> '
        f"= dense<{_dense_2d(patches0)}>")
    if bias_global:
        lines.append(bias_global.rstrip("\n"))
    lines.append(f"func.func private @{print_call}({unranked_type})")
    lines.append("func.func @main() {")
    for sym, w in zip(weight_syms, weights2d):
        cout, k = w.shape
        lines.append(f"  %{sym} = memref.get_global @{sym} : memref<{cout}x{k}xi8>")
    lines.append(f"  %aInit = memref.get_global @{act_sym} : memref<{m0}x{k0}xi8>")
    lines.append(f"  %a0 = memref.alloc() : memref<{m0}x{k0}xi8>")
    lines.append(
        f"  memref.copy %aInit, %a0 : memref<{m0}x{k0}xi8> to memref<{m0}x{k0}xi8>")
    # One shared zero constant for every layer's zero-fill (linalg.fill
    # takes a scalar, not a splat attribute) -- hoisted once rather than
    # redeclared per layer, since the value is always the same.
    lines.append("  %zeroI8 = arith.constant 0 : i8")
    # Same idea for a pool's own fill value -- INT8_MIN, so a padded
    # position never wins a max (see this function's own `pool_params`
    # docstring) -- but only declared when at least one pool actually
    # runs, so a chain with no pooling emits byte-identical text to
    # before this parameter existed.
    if pool_params:
        lines.append("  %negI8 = arith.constant -128 : i8")

    act_val = "%a0"
    cur_m, cur_k = m0, k0
    cur_h, cur_w = out_h, out_w
    tmp_counter = [0]

    def fresh(prefix):
        tmp_counter[0] += 1
        return f"%{prefix}{tmp_counter[0]}"

    for i, (sym, w) in enumerate(zip(weight_syms, weights2d)):
        cout, k = w.shape
        out_val = f"%out{i}"
        lines.append(f"  {out_val} = memref.alloc() : memref<{cur_m}x{cout}xi32>")
        lines.append(
            f"  linalg.matmul_transpose_b ins({act_val}, %{sym} : "
            f"memref<{cur_m}x{cur_k}xi8>, memref<{cout}x{k}xi8>)")
        lines.append(f"    outs({out_val} : memref<{cur_m}x{cout}xi32>)")

        if i == n_layers - 1:
            if ends_in_conv:
                # Apply the same bias/requantize tail emit_module applies
                # to its own single conv, precomputed above (needed
                # earlier for the bias global's own module-scope
                # placement).
                lines += tail_lines
            else:
                # A matmul-terminated chain: this layer's own raw
                # accumulator IS the printed result, exactly
                # emit_chain_module's own convention -- only known now,
                # since `cur_m` may have changed during the conv portion.
                result_val = out_val
                result_type = f"memref<{cur_m}x{cout}xi32>"
            break

        bridged = f"%act{i + 1}"
        lines.append(
            f"  {bridged} = cim.requantize {out_val} {{scale = "
            f"{float(scales[i])!r} : f32, zero_point = 0 : i32, "
            f"effective_bits = 8 : i32}}")
        lines.append(f"    : memref<{cur_m}x{cout}xi32> -> memref<{cur_m}x{cout}xi8>")

        if i in pool_params:
            # A MaxPool between this bridge and layer i+1's own gather --
            # see this function's own `pool_params` docstring for the full
            # design. Reassigns `bridged`/`cur_m`/`cur_h`/`cur_w` in
            # place, so everything below (the matmul branch, and the
            # conv-gather branch's own expand_shape) sees the POOLED
            # activation and its own, already-updated spatial size,
            # exactly as if layer i's bridge had produced it directly.
            (pkh, pkw, p_stride_h, p_stride_w, p_pad_top, p_pad_bottom,
             p_pad_left, p_pad_right, p_dilation_h, p_dilation_w) = (
                pool_params[i])
            pool_cin = cout  # pooling preserves channel count.

            pexp_ty = f"memref<{n_batch}x{cur_h}x{cur_w}x{pool_cin}xi8>"
            pexp = fresh("poolExp")
            lines.append(
                f"  {pexp} = memref.expand_shape {bridged} [[0, 1, 2], "
                f"[3]] : memref<{cur_m}x{pool_cin}xi8> into {pexp_ty}")

            pool_out_h = output_size(cur_h, pkh, p_pad_top, p_pad_bottom,
                                     p_stride_h, p_dilation_h)
            pool_out_w = output_size(cur_w, pkw, p_pad_left, p_pad_right,
                                     p_stride_w, p_dilation_w)
            php = cur_h + p_pad_top + p_pad_bottom
            pwp = cur_w + p_pad_left + p_pad_right

            ppadded_ty = f"memref<{n_batch}x{php}x{pwp}x{pool_cin}xi8>"
            ppadded = fresh("poolPadded")
            lines.append(f"  {ppadded} = memref.alloc() : {ppadded_ty}")
            lines.append(
                f"  linalg.fill ins(%negI8 : i8) outs({ppadded} : "
                f"{ppadded_ty})")

            pinner_stride = [php * pwp * pool_cin, pwp * pool_cin,
                             pool_cin, 1]
            pinner_off = p_pad_top * pwp * pool_cin + p_pad_left * pool_cin
            pinner_ty = (f"memref<{n_batch}x{cur_h}x{cur_w}x{pool_cin}xi8, "
                        f"strided<{pinner_stride}, offset: {pinner_off}>>")
            pinner = fresh("poolPaddedInner")
            lines.append(
                f"  {pinner} = memref.subview {ppadded}[0, {p_pad_top}, "
                f"{p_pad_left}, 0] [{n_batch}, {cur_h}, {cur_w}, "
                f"{pool_cin}] [1, 1, 1, 1] : {ppadded_ty} to {pinner_ty}")
            lines.append(
                f"  memref.copy {pexp}, {pinner} : {pexp_ty} to {pinner_ty}")

            pool_m = n_batch * pool_out_h * pool_out_w
            pooled4d_ty = (f"memref<{n_batch}x{pool_out_h}x{pool_out_w}x"
                          f"{pool_cin}xi8>")
            tap_operands = []
            tap_types = []
            for th in range(pkh):
                for tw in range(pkw):
                    h_off = th * p_dilation_h
                    w_off = tw * p_dilation_w
                    tap_stride = [php * pwp * pool_cin,
                                 pwp * pool_cin * p_stride_h,
                                 pool_cin * p_stride_w, 1]
                    tap_off = h_off * pwp * pool_cin + w_off * pool_cin
                    tap_ty = (f"memref<{n_batch}x{pool_out_h}x"
                             f"{pool_out_w}x{pool_cin}xi8, "
                             f"strided<{tap_stride}, offset: {tap_off}>>")
                    tap = fresh("poolTap")
                    lines.append(
                        f"  {tap} = memref.subview {ppadded}[0, {h_off}, "
                        f"{w_off}, 0] [{n_batch}, {pool_out_h}, "
                        f"{pool_out_w}, {pool_cin}] [1, {p_stride_h}, "
                        f"{p_stride_w}, 1] : {ppadded_ty} to {tap_ty}")
                    tap_operands.append(tap)
                    tap_types.append(tap_ty)

            pooled = fresh("poolMax")
            lines.append(
                f"  {pooled} = cim.reduce_max {', '.join(tap_operands)} : "
                f"({', '.join(tap_types)}) -> {pooled4d_ty}")

            pooled2d_ty = f"memref<{pool_m}x{pool_cin}xi8>"
            pooled2d = fresh("poolCollapsed")
            lines.append(
                f"  {pooled2d} = memref.collapse_shape {pooled} "
                f"[[0, 1, 2], [3]] : {pooled4d_ty} into {pooled2d_ty}")

            bridged = pooled2d
            cur_m = pool_m
            cur_h, cur_w = pool_out_h, pool_out_w

        if i + 1 >= n_conv_layers:
            # A plain MatMulInteger layer: the bridged activation is
            # already the flat [M, K] shape a matmul needs, no gather at
            # all -- exactly emit_chain_module's own inter-matmul step.
            # M threads through unchanged (that function's own note).
            act_val = bridged
            cur_k = cout
            continue

        (kh, kw, stride_h, stride_w, pad_top, pad_bottom, pad_left,
         pad_right, dilation_h, dilation_w) = conv_params[i + 1]
        cin = cout  # layer i+1's Cin is layer i's own Cout.
        h, w_ = cur_h, cur_w
        dilated_kh = (kh - 1) * dilation_h + 1
        dilated_kw = (kw - 1) * dilation_w + 1
        hp = h + pad_top + pad_bottom
        wp = w_ + pad_left + pad_right
        next_m, next_out_h, next_out_w = layer_shapes[i + 1]
        next_k = kh * kw * cin

        exp_ty = f"memref<{n_batch}x{h}x{w_}x{cin}xi8>"
        exp = fresh("aExp")
        lines.append(
            f"  {exp} = memref.expand_shape {bridged} [[0, 1, 2], [3]] : "
            f"memref<{cur_m}x{cin}xi8> into {exp_ty}")

        padded_ty = f"memref<{n_batch}x{hp}x{wp}x{cin}xi8>"
        padded = fresh("padded")
        lines.append(f"  {padded} = memref.alloc() : {padded_ty}")
        lines.append(f"  linalg.fill ins(%zeroI8 : i8) outs({padded} : {padded_ty})")

        inner_stride = [hp * wp * cin, wp * cin, cin, 1]
        inner_off = pad_top * wp * cin + pad_left * cin
        inner_ty = (f"memref<{n_batch}x{h}x{w_}x{cin}xi8, "
                   f"strided<{inner_stride}, offset: {inner_off}>>")
        inner = fresh("paddedInner")
        lines.append(
            f"  {inner} = memref.subview {padded}[0, {pad_top}, {pad_left}, 0] "
            f"[{n_batch}, {h}, {w_}, {cin}] [1, 1, 1, 1] : {padded_ty} to {inner_ty}")
        lines.append(f"  memref.copy {exp}, {inner} : {exp_ty} to {inner_ty}")

        patches = fresh("patches")
        patches_ty = f"memref<{next_m}x{next_k}xi8>"
        lines.append(f"  {patches} = memref.alloc() : {patches_ty}")

        for th in range(kh):
            for tw in range(kw):
                t = th * kw + tw
                h_off = th * dilation_h
                w_off = tw * dilation_w
                tap_stride = [hp * wp * cin, wp * cin * stride_h,
                              cin * stride_w, 1]
                tap_off = h_off * wp * cin + w_off * cin
                tap_ty = (f"memref<{n_batch}x{next_out_h}x{next_out_w}x{cin}xi8, "
                         f"strided<{tap_stride}, offset: {tap_off}>>")
                tap = fresh("tap")
                lines.append(
                    f"  {tap} = memref.subview {padded}[0, {h_off}, {w_off}, 0] "
                    f"[{n_batch}, {next_out_h}, {next_out_w}, {cin}] "
                    f"[1, {stride_h}, {stride_w}, 1] : {padded_ty} to {tap_ty}")

                slab_ty = f"memref<{n_batch}x{next_out_h}x{next_out_w}x{cin}xi8>"
                slab = fresh("tapSlab")
                lines.append(f"  {slab} = memref.alloc() : {slab_ty}")
                lines.append(f"  memref.copy {tap}, {slab} : {tap_ty} to {slab_ty}")

                coll_ty = f"memref<{next_m}x{cin}xi8>"
                coll = fresh("tapCollapsed")
                lines.append(
                    f"  {coll} = memref.collapse_shape {slab} [[0, 1, 2], [3]] "
                    f": {slab_ty} into {coll_ty}")

                col_stride = [next_k, 1]
                col_off = t * cin
                col_ty = (f"memref<{next_m}x{cin}xi8, "
                         f"strided<{col_stride}, offset: {col_off}>>")
                col = fresh("patchesCol")
                lines.append(
                    f"  {col} = memref.subview {patches}[0, {col_off}] "
                    f"[{next_m}, {cin}] [1, 1] : {patches_ty} to {col_ty}")
                lines.append(f"  memref.copy {coll}, {col} : {coll_ty} to {col_ty}")

        act_val = patches
        cur_m, cur_k = next_m, next_k
        cur_h, cur_w = next_out_h, next_out_w

    lines.append(f"  %u = memref.cast {result_val} : {result_type} to {unranked_type}")
    lines.append(f"  func.call @{print_call}(%u) : ({unranked_type}) -> ()")
    lines.append(f"  memref.dealloc %a0 : memref<{m0}x{k0}xi8>")
    lines.append(f"  memref.dealloc {result_val} : {result_type}")
    lines.append("  return")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
