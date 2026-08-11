"""Read an ONNX model; produce the MLIR the cim pipeline consumes.

WHY MatMulInteger AND NOTHING ELSE
==================================
ONNX `MatMulInteger` takes int8 A [M, K] and int8 B [K, N] and produces
int32 Y [M, N]. That is exactly the v0.1 contract in a single op -- "INT8
in, wider integer accumulator out" (lib/Transforms/CIMDetect.cpp) -- so it
needs no quantization modelling at all to reach.

The neighbouring ops are refused, and each for a reason rather than for
lack of time:

  * `MatMul` + QuantizeLinear/DequantizeLinear is float arithmetic. To
    offload it the importer would have to choose scales, i.e. perform
    calibration. lib/Transforms/CIMLegalizePrecision.cpp refuses to invent
    calibration data for exactly this reason and so does this.
  * `Gemm` carries alpha/beta and a bias operand C. The cim dialect has no
    bias op, so supporting Gemm means either dropping C or fusing it, and
    both silently change the arithmetic.
  * `QLinearMatMul` produces a quantized int8 result, which reintroduces
    the rounding-mode divergence described below.

THE TRANSPOSE, WHICH IS THE DANGEROUS PART
==========================================
ONNX computes Y[m][n] = sum_k A[m][k] * B[k][n], so B is [K, N].
`cimrt_mvm` (runtime/src/simulator/simulator.cpp) computes
out[r] = sum_c W[r*cols + c] * act[c], so W is [N, K]. Therefore W = B.T,
and this module is the ONE place that flip happens.

Getting it backwards does not fail: it emits structurally perfect IR that
passes every verifier and computes wrong numbers -- and on a square weight
it is completely silent. test/python/test_onnx_frontend.py guards it three
ways (a direct assertion on the emitted literal, a mutation check proving
that assertion has teeth, and rectangular shapes throughout so a missing
transpose is a loud shape error rather than a quiet wrong answer).

WHY THE ACTIVATION MUST BE SUPPLIED
===================================
The emitted module is `func.func @main()` with no arguments -- cim-run has
no mechanism to pass data in, so the module is one baked inference and the
activation is a constant. Defaulting it to zeros would be catastrophic
rather than merely lazy: 0 @ W == 0 for every W, so the differential test
would pass against any weight matrix, transposed or not, and the single
largest risk in this front end would be invisible. So it is required, in
the same way cim-partition requires -target-yaml rather than guessing tile
geometry.
"""

import hashlib

import numpy as np

from .emit import emit_chain_module, emit_module, sanitize_symbol
from .im2col import im2col_nchw
from .refusal import Refusal

# MatMulInteger was introduced at opset 10.
MIN_OPSET = 10

ACCEPTED_OP = "MatMulInteger"

# QLinearConv was introduced at the same opset.
ACCEPTED_CONV_OP = "QLinearConv"


def _tensor_names(model):
    """Names of every initializer, i.e. every compile-time-constant tensor."""
    return {init.name for init in model.graph.initializer}


def _check_opset(model):
    for imp in model.opset_import:
        # The default (empty) domain is the one MatMulInteger lives in.
        if imp.domain in ("", "ai.onnx"):
            if imp.version < MIN_OPSET:
                raise Refusal(
                    f"model targets opset {imp.version}, but MatMulInteger was "
                    f"introduced at opset {MIN_OPSET}. Nothing emitted.")
            return
    raise Refusal("model declares no default-domain opset import; cannot tell "
                  "whether MatMulInteger is available. Nothing emitted.")


def _shape_of(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_param") or not d.HasField("dim_value") or d.dim_value <= 0:
            return None  # symbolic or unknown
        dims.append(d.dim_value)
    return dims


def _zero_point_is_zero(model, name, node_name, which):
    """A zero point operand must be absent or all zero."""
    from onnx import numpy_helper

    if not name:
        return
    for init in model.graph.initializer:
        if init.name == name:
            values = numpy_helper.to_array(init)
            if np.any(values != 0):
                raise Refusal(
                    f"{which} is {values.ravel().tolist()}, but the v0.1 "
                    f"contract is symmetric int8 (zero point 0). Folding a "
                    f"non-zero b_zero_point into the constant weights is only "
                    f"valid while every element of (B - zp) stays in int8, and "
                    f"a non-zero a_zero_point additionally needs a per-output "
                    f"bias term, which this dialect has no op for. "
                    f"Nothing emitted.",
                    where=f"node '{node_name}'")
            return
    raise Refusal(
        f"{which} '{name}' is not a constant initializer, so its value cannot "
        f"be checked at compile time. Nothing emitted.",
        where=f"node '{node_name}'")


def load_matmul_integer(model):
    """The single accepted MatMulInteger node, as (node, weight[N,K], act_name).

    Raises Refusal for anything outside the v0.1 contract. The weight comes
    back ALREADY transposed into cim.mvm's output-major [N, K] convention.
    """
    from onnx import numpy_helper

    _check_opset(model)

    graph = model.graph
    matmuls = [n for n in graph.node if n.op_type == ACCEPTED_OP]
    if not matmuls:
        seen = sorted({n.op_type for n in graph.node})
        raise Refusal(
            f"no {ACCEPTED_OP} node in the graph (found: "
            f"{', '.join(seen) if seen else 'no nodes at all'}). This front "
            f"end offloads {ACCEPTED_OP} only. Nothing emitted.")
    if len(matmuls) > 1:
        raise Refusal(
            f"found {len(matmuls)} {ACCEPTED_OP} nodes; load_matmul_integer "
            f"only handles a single one. import_model() dispatches to "
            f"load_matmul_chain() automatically for a multi-layer graph -- "
            f"call that instead, or call import_model() directly rather than "
            f"this function. Nothing emitted.")

    node = matmuls[0]
    where = f"node '{node.name or ACCEPTED_OP}'"

    # Every OTHER node is refused rather than skipped: silently dropping a
    # Relu or an Add emits a module that compiles and computes a different
    # function than the model does.
    others = [n for n in graph.node if n is not node]
    if others:
        kinds = sorted({n.op_type for n in others})
        raise Refusal(
            f"graph also contains {', '.join(kinds)}. This front end imports a "
            f"graph consisting of one {ACCEPTED_OP} and nothing else -- "
            f"ignoring an op it does not understand would emit a module that "
            f"computes something other than the model. Nothing emitted.")

    initializers = _tensor_names(model)
    a_name, b_name = node.input[0], node.input[1]

    if b_name not in initializers:
        raise Refusal(
            f"the weight operand '{b_name}' is not a constant initializer. "
            f"Weight-stationary hardware has nothing to make resident if the "
            f"weights are computed at runtime. Nothing emitted.", where=where)
    if a_name in initializers:
        raise Refusal(
            f"both operands are constant initializers. cim-detect requires "
            f"exactly one constant operand -- a matmul of two constants has "
            f"nothing to make resident, and would silently not be offloaded. "
            f"Nothing emitted.", where=where)

    # zero_point operands are inputs 2 (a) and 3 (b), both optional.
    a_zp = node.input[2] if len(node.input) > 2 else ""
    b_zp = node.input[3] if len(node.input) > 3 else ""
    _zero_point_is_zero(model, a_zp, node.name or ACCEPTED_OP, "a_zero_point")
    _zero_point_is_zero(model, b_zp, node.name or ACCEPTED_OP, "b_zero_point")

    b_init = next(i for i in graph.initializer if i.name == b_name)
    # numpy_helper handles the raw_data/int32_data field split and the
    # little-endian row-major layout. A hand-rolled np.frombuffer does not,
    # and would silently misread whichever field this model happens to use.
    weight = numpy_helper.to_array(b_init)

    if weight.dtype != np.int8:
        raise Refusal(
            f"the weight operand '{b_name}' has dtype {weight.dtype}. "
            f"MatMulInteger permits uint8, but cim.mvm and the simulator are "
            f"signed throughout: reinterpreting a uint8 200 as int8 would "
            f"silently compute with -56. Nothing emitted.", where=where)
    if weight.ndim != 2:
        raise Refusal(
            f"the weight operand '{b_name}' has rank {weight.ndim}; "
            f"cim-partition requires rank-2 operands. Nothing emitted.",
            where=where)

    a_info = next((v for v in graph.input if v.name == a_name), None)
    if a_info is None:
        raise Refusal(
            f"the activation operand '{a_name}' is neither a graph input nor a "
            f"constant. Nothing emitted.", where=where)
    if a_info.type.tensor_type.elem_type != 3:  # TensorProto.INT8
        raise Refusal(
            f"the activation operand '{a_name}' is not int8. The v0.1 contract "
            f"is INT8 in, int32 accumulator out. Nothing emitted.", where=where)

    a_shape = _shape_of(a_info)
    if a_shape is None:
        raise Refusal(
            f"the activation operand '{a_name}' has a symbolic or unknown "
            f"shape. Weights and activations are materialized as dense "
            f"literals, so every dimension must be a known constant. Nothing "
            f"emitted.", where=where)
    if len(a_shape) != 2:
        raise Refusal(
            f"the activation operand '{a_name}' has rank {len(a_shape)}; "
            f"cim-partition requires rank-2 operands. Nothing emitted.",
            where=where)

    m, k = a_shape
    if weight.shape[0] != k:
        raise Refusal(
            f"shape mismatch: activation is [{m}, {k}] but the weight is "
            f"{list(weight.shape)}; MatMulInteger needs B to be [K, N] with "
            f"K = {k}. Nothing emitted.", where=where)

    # THE transpose. ONNX B is [K, N]; cim.mvm wants output-major [N, K].
    return node, np.ascontiguousarray(weight.T), a_name


# ---------------------------------------------------------------------------
# Chained layers: MatMulInteger -> Cast(to=float32) -> QuantizeLinear(scale, 0)
# -> MatMulInteger -> ...
#
# WHY THIS EXACT BRIDGE, AND NOTHING ELSE
# ========================================
# Between two offloaded matmuls, layer i's int32 accumulator has to become
# layer i+1's int8 activation -- something has to narrow it, and getting
# that narrowing wrong is a second, quieter place a wrong answer could
# hide (the first being the transpose above).
#
# The bridge's scale may be ANY positive constant, not just 1.0 -- a real,
# calibrated per-layer scale, the standard shape of a real post-training-
# quantized multi-layer INT8 model. cim.requantize's own arithmetic is
# already scale/zero_point/effective_bits-generic (test/mlir/
# legalize_precision_e2e_test.cpp proves it at scale=2.0), so there was
# never a machinery reason to require 1.0 here -- only a testing-
# convenience one, explained below.
#
# What scale=1.0 specifically buys, and a real scale does not:
#
#   1. cim.requantize (what the compiled side actually executes, emitted
#      by emit.emit_chain_module) rounds half-away-from-zero.
#      QuantizeLinear (what ONNX defines) rounds half-to-even. Those two
#      modes diverge exactly at a tie -- a value ending in exactly .5 --
#      and NOWHERE else. With scale == 1.0, both sides compute
#      round(v / 1.0) on a v that is ALREADY an integer, so there is never
#      a fractional part and therefore never a tie to break differently:
#      the compiled result matches an UNQUANTIZED ONNX oracle exactly. A
#      real scale can land exactly on a tie -- a real, hardware-fidelity
#      fact about cim.requantize's modeled rounding mode (a digital
#      requantizer or ADC readout path), not a bug -- so a real-scale
#      chain is compared against onnx.reference's own full (quantized)
#      evaluation instead, in test_onnx_frontend_chain.py, which documents
#      the exact-tie divergence explicitly rather than treating it as a
#      mysterious near-miss.
#   2. Both sides then saturate to the same range regardless of scale:
#      cim.requantize with effective_bits=8 clamps to [-128, 127], and
#      QuantizeLinear with an explicit int8 zero_point saturates to the
#      same range by definition.
#
# scale=1.0's exactness was checked against a real cim-opt/cim-run round
# trip (--cim-detect --cim-partition --cim-placement, deliberately NOT
# --cim-legalize-precision -- see this file's own module docstring) before
# any of this graph-walking code was written: a hand-written 2-layer
# chain's compiled output matched `np.clip(layer0_output, -128,
# 127).astype(np.int8)` fed into layer 1, exactly.
#
# Any other rounding path, a non-constant scale, or zero_point left absent
# or non-zero (which either makes QuantizeLinear's own output dtype
# ambiguous -- uint8 by default in the opsets this checks -- or asks for
# an asymmetric zero point this dialect cannot express) is still refused
# rather than approximated.
# ---------------------------------------------------------------------------

_CAST_TO_FLOAT = 1  # TensorProto.FLOAT


def _initializer_array(graph, name):
    from onnx import numpy_helper

    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def _validate_bridge(graph, producer, consumers, expected_source, act_name,
                     where):
    """`act_name` must be `expected_source` narrowed by exactly one
    Cast(to=float32) -> QuantizeLinear(scale=<positive>, zero_point=0 int8)
    pair, consumed by nothing else along the way. Returns the bridge's
    scale (a Python float) on success; raises Refusal otherwise.

    Any positive scale is accepted, not just 1.0: cim.requantize's
    arithmetic is scale/zero_point/effective_bits-generic already (see
    test/mlir/legalize_precision_e2e_test.cpp), so refusing a real,
    calibrated scale here would be refusing a caller-supplied number this
    project's own compiler has no trouble with. What a non-1.0 scale
    changes is exactness against an UNQUANTIZED oracle: QuantizeLinear
    (round-half-to-even) and cim.requantize (round-half-away-from-zero)
    can only diverge at a rounding tie, and scale=1.0 is what makes
    `v / scale` always already an integer, i.e. never a tie to round
    differently. A real scale can land exactly on a tie -- a real,
    hardware-fidelity fact about cim.requantize's modeled rounding mode,
    not a bug -- so the emitted module compiled from a real scale should
    be compared against onnx.reference's own full quantized evaluation,
    not an unquantized one. See emit.emit_chain_module's own docstring.
    """
    quant_node = producer.get(act_name)
    if quant_node is None or quant_node.op_type != "QuantizeLinear":
        raise Refusal(
            f"the activation operand '{act_name}' is not produced by "
            f"QuantizeLinear. Layers in a chain may only be bridged by "
            f"Cast(to=float32) -> QuantizeLinear(scale=<positive>, "
            f"zero_point=0) -- anything else risks a rounding-mode or "
            f"overflow divergence from what actually gets compiled. "
            f"Nothing emitted.", where=where)

    scale_name = quant_node.input[1] if len(quant_node.input) > 1 else ""
    scale_arr = _initializer_array(graph, scale_name)
    scale = None if scale_arr is None else float(np.asarray(scale_arr).reshape(-1)[0])
    if scale is None or not (scale > 0.0):
        raise Refusal(
            f"QuantizeLinear '{quant_node.name}' has scale "
            f"{scale_arr.tolist() if scale_arr is not None else '<non-constant>'}"
            f"; a bridge scale must be a positive constant. Nothing "
            f"emitted.", where=where)

    zp_name = quant_node.input[2] if len(quant_node.input) > 2 else ""
    zp_arr = _initializer_array(graph, zp_name)
    if zp_arr is None:
        raise Refusal(
            f"QuantizeLinear '{quant_node.name}' has no zero_point operand. "
            f"An explicit int8 zero_point of 0 is required -- it is also "
            f"what fixes the output dtype at int8 rather than ONNX's default "
            f"of uint8. Nothing emitted.", where=where)
    if np.asarray(zp_arr).dtype != np.int8:
        raise Refusal(
            f"QuantizeLinear '{quant_node.name}' has a "
            f"{np.asarray(zp_arr).dtype} zero_point; cim.mvm and the "
            f"simulator are signed throughout, so the bridge must produce "
            f"int8. Nothing emitted.", where=where)
    if np.any(np.asarray(zp_arr) != 0):
        raise Refusal(
            f"QuantizeLinear '{quant_node.name}' has a non-zero zero_point "
            f"({zp_arr.tolist()}). The v0.1 contract is symmetric int8. "
            f"Nothing emitted.", where=where)

    cast_name = quant_node.input[0]
    cast_node = producer.get(cast_name)
    if cast_node is None or cast_node.op_type != "Cast":
        raise Refusal(
            f"QuantizeLinear '{quant_node.name}' does not read directly from "
            f"a Cast. Nothing emitted.", where=where)
    to_attr = next((a for a in cast_node.attribute if a.name == "to"), None)
    if to_attr is None or to_attr.i != _CAST_TO_FLOAT:
        raise Refusal(
            f"Cast '{cast_node.name}' targets a type other than float32. "
            f"Casting an int32 accumulator to float32 and back with scale=1.0 "
            f"is exact for |value| < 2**24, which is what makes this bridge "
            f"safe to compare against an unquantized ONNX oracle; any other "
            f"target type has not been checked for that property. Nothing "
            f"emitted.", where=where)
    if cast_node.input[0] != expected_source:
        raise Refusal(
            f"Cast '{cast_node.name}' does not read the previous layer's own "
            f"output. Nothing emitted.", where=where)

    # No fan-out along the bridge: the chain this front end imports is
    # strictly linear, and a DAG needs buffer-liveness reasoning this
    # importer does not do. This covers all three tensors the bridge
    # touches -- the raw accumulator, the intermediate float cast, and the
    # bridge's own int8 output -- because a stray extra reader of any one
    # of them is equally a DAG this importer cannot safely reason about.
    for tensor, label in ((expected_source, "the previous layer's output"),
                          (cast_name, f"Cast '{cast_node.name}'s output"),
                          (act_name, f"QuantizeLinear '{quant_node.name}'s "
                                     f"output")):
        readers = consumers.get(tensor, [])
        if len(readers) != 1:
            raise Refusal(
                f"{label} is read by {len(readers)} node(s); this front end "
                f"imports a strictly linear chain, and a value read more than "
                f"once needs buffer-liveness reasoning it does not do. "
                f"Nothing emitted.", where=where)

    return scale


def load_matmul_chain(model):
    """Two or more MatMulInteger nodes bridged as described above.

    Returns (nodes, weights, first_activation_name, scales), where
    `weights[i]` is layer i's weight ALREADY transposed to cim.mvm's
    [N, K] convention, and `scales[i]` is bridge i's real, positive
    QuantizeLinear scale (see _validate_bridge's own docstring).
    """
    from onnx import numpy_helper

    _check_opset(model)
    graph = model.graph
    initializers = _tensor_names(model)

    matmul_nodes = [n for n in graph.node if n.op_type == ACCEPTED_OP]
    if len(matmul_nodes) < 2:
        raise Refusal(
            "load_matmul_chain requires at least two MatMulInteger nodes; "
            "use load_matmul_integer for a single layer.")

    allowed = {ACCEPTED_OP, "Cast", "QuantizeLinear"}
    others = [n for n in graph.node if n.op_type not in allowed]
    if others:
        kinds = sorted({n.op_type for n in others})
        raise Refusal(
            f"graph also contains {', '.join(kinds)}. A chain of "
            f"{ACCEPTED_OP} nodes may only be bridged by Cast(to=float32) -> "
            f"QuantizeLinear(scale=1.0, zero_point=0) -- ignoring any other "
            f"op would emit a module that computes something other than the "
            f"model. Nothing emitted.")

    if len(graph.output) != 1:
        raise Refusal(
            f"graph has {len(graph.output)} outputs; a chain must end in "
            f"exactly one. Nothing emitted.")
    if graph.output[0].name != matmul_nodes[-1].output[0]:
        raise Refusal(
            f"the graph's output is not the last {ACCEPTED_OP}'s own output "
            f"-- only a chain that ends directly at the last layer's int32 "
            f"accumulator, with no trailing op, is supported. Nothing "
            f"emitted.")

    producer = {out: n for n in graph.node for out in n.output}
    consumers = {}
    for n in graph.node:
        for inp in n.input:
            if inp:
                consumers.setdefault(inp, []).append(n)

    weights = []
    scales = []
    first_act_name = None
    for i, node in enumerate(matmul_nodes):
        where = f"node '{node.name or ACCEPTED_OP}' (layer {i})"
        a_name, b_name = node.input[0], node.input[1]

        if b_name not in initializers:
            raise Refusal(
                f"the weight operand '{b_name}' is not a constant "
                f"initializer. Nothing emitted.", where=where)
        if a_name in initializers:
            raise Refusal(
                f"both operands are constant initializers. Nothing emitted.",
                where=where)

        a_zp = node.input[2] if len(node.input) > 2 else ""
        b_zp = node.input[3] if len(node.input) > 3 else ""
        _zero_point_is_zero(model, a_zp, node.name or ACCEPTED_OP,
                            "a_zero_point")
        _zero_point_is_zero(model, b_zp, node.name or ACCEPTED_OP,
                            "b_zero_point")

        weight = numpy_helper.to_array(
            next(t for t in graph.initializer if t.name == b_name))
        if weight.dtype != np.int8:
            raise Refusal(
                f"the weight operand '{b_name}' has dtype {weight.dtype}; "
                f"cim.mvm and the simulator are signed throughout. Nothing "
                f"emitted.", where=where)
        if weight.ndim != 2:
            raise Refusal(
                f"the weight operand '{b_name}' has rank {weight.ndim}; "
                f"cim-partition requires rank-2 operands. Nothing emitted.",
                where=where)

        if i == 0:
            a_info = next((v for v in graph.input if v.name == a_name), None)
            if a_info is None:
                raise Refusal(
                    f"the activation operand '{a_name}' is neither a graph "
                    f"input nor a constant. Nothing emitted.", where=where)
            if a_info.type.tensor_type.elem_type != 3:  # TensorProto.INT8
                raise Refusal(
                    f"the activation operand '{a_name}' is not int8. Nothing "
                    f"emitted.", where=where)
            a_shape = _shape_of(a_info)
            if a_shape is None:
                raise Refusal(
                    f"the activation operand '{a_name}' has a symbolic or "
                    f"unknown shape. Nothing emitted.", where=where)
            if len(a_shape) != 2:
                raise Refusal(
                    f"the activation operand '{a_name}' has rank "
                    f"{len(a_shape)}; cim-partition requires rank-2 "
                    f"operands. Nothing emitted.", where=where)
            if weight.shape[0] != a_shape[1]:
                raise Refusal(
                    f"shape mismatch: activation is {a_shape} but the weight "
                    f"is {list(weight.shape)}. Nothing emitted.", where=where)
            first_act_name = a_name
        else:
            scales.append(_validate_bridge(
                graph, producer, consumers, matmul_nodes[i - 1].output[0],
                a_name, where))
            if weight.shape[0] != weights[-1].shape[0]:
                raise Refusal(
                    f"layer {i}'s K ({weight.shape[0]}) does not match layer "
                    f"{i - 1}'s N ({weights[-1].shape[0]}). Nothing emitted.",
                    where=where)

        # THE transpose, same as the single-layer path.
        weights.append(np.ascontiguousarray(weight.T))

    return matmul_nodes, weights, first_act_name, scales


# ---------------------------------------------------------------------------
# A single QLinearConv node -- a 2-D convolution IS a matmul, once
#
# WHY THIS IS TRACTABLE AT ALL
# =============================
# A general convolution is genuinely a different primitive than a
# matrix-vector multiply, and turning one into the other (im2col) is
# usually considered expensive: it materializes every overlapping window
# of the activation, redundantly, at every inference. That cost does not
# apply here. This file's own module docstring already establishes that
# every emitted module is ONE BAKED INFERENCE -- cim-run has no mechanism
# to pass data in, so the activation is always a compile-time constant.
# im2col over a constant, at import time, in Python, happens exactly once,
# ever, for a given (model, activation) pair -- there is no redundant
# per-inference cost to worry about, because there is no per-inference
# cost at all in this project's execution model. See im2col.py's own
# module docstring for the reshape that makes this work with NO new MLIR
# op and NO change to cim-detect/cim-partition/the interpreter: this
# function's whole job is producing the same (weight[N, K], activation
# [M, K]) shape load_matmul_integer already does, so emit.emit_module can
# be reused completely unchanged.
#
# WHAT IS DELIBERATELY REFUSED, AND WHY
# =======================================
# QLinearConv is a much larger op than MatMulInteger -- grouped/depthwise
# convolution, dilation, per-output-channel (per-axis) quantization, and a
# fused int32 bias are all real, common features of quantized conv models
# that a production quantization toolchain routinely produces. Each is
# refused explicitly rather than approximated:
#
#   * group != 1: each group is really an independent matmul over a slice
#     of channels, not one reshape. Silently computing group=1's answer
#     for a grouped (e.g. depthwise) conv would be a confident wrong
#     number, not a shape error.
#   * dilations != (1, 1): a dilated kernel tap reads a non-contiguous
#     patch; im2col_nchw's contiguous-slice implementation does not
#     express that, and pretending stride covers it would too.
#   * a non-scalar w_scale (per-output-channel quantization): a real
#     `cim.requantize` call takes exactly one scalar `scale` attribute, so
#     per-channel scales cannot be expressed as one op. Averaging or
#     picking one channel's scale would silently change the arithmetic
#     for every other channel.
#   * a non-zero x_zero_point, w_zero_point, or y_zero_point: the same
#     "needs a per-output bias term this dialect cannot express" reason
#     load_matmul_integer's own _zero_point_is_zero refuses it for.
#   * a bias operand B: QLinearConv adds B directly to the raw int32
#     accumulator, per-output-channel, before scaling. There is no cim op
#     that adds two same-shape int32 buffers at THIS point in the
#     pipeline (cim.reduce_partial exists, but only as an implementation
#     detail cim-partition inserts for its own K-tiling, after this
#     import-time code has already run) without risking an interaction
#     with that pass this code has not verified -- the same discipline
#     `Gemm`'s bias operand is refused for above.
#   * `auto_pad` other than "NOTSET" (explicit `pads`): SAME_UPPER/
#     SAME_LOWER/VALID all need shape-dependent padding math to replicate
#     exactly, and getting that arithmetic wrong is a silent one-pixel
#     offset, not a crash. VALID specifically (which the ONNX spec defines
#     as simply "no padding") is refused too, rather than special-cased as
#     the trivial pads=[0, 0, 0, 0] it should be: onnx.reference's own
#     Conv implementation (op_conv._conv_implementation) computes VALID's
#     padding with the same shape-dependent formula as SAME_UPPER/
#     SAME_LOWER, and that formula indexes `X.shape[i]` for the spatial
#     dims without skipping the leading N/C axes -- so it is wrong for any
#     model with N != 1 or C != 1, which is most of them. That is a bug in
#     the oracle, not something this front end should route around
#     silently; refusing all `auto_pad` but NOTSET sidesteps it entirely,
#     and costs nothing real, since NOTSET with `pads=[0, 0, 0, 0]`
#     (also this function's own default) is already exactly VALID's
#     defined behavior.
# ---------------------------------------------------------------------------

_INT8 = 3  # TensorProto.INT8


def _initializer_or_refuse(graph, name, node_name, which):
    arr = _initializer_array(graph, name)
    if arr is None:
        raise Refusal(
            f"{which} '{name}' is not a constant initializer, so its value "
            f"cannot be checked at compile time. Nothing emitted.",
            where=f"node '{node_name}'")
    return arr


def _positive_scalar_scale(graph, name, node_name, which):
    arr = np.asarray(_initializer_or_refuse(graph, name, node_name, which))
    if arr.size != 1:
        raise Refusal(
            f"{which} has {arr.size} elements -- per-output-channel "
            f"(per-axis) quantization. A single cim.requantize call takes "
            f"exactly one scalar scale; picking or averaging one channel's "
            f"scale would silently change the arithmetic for every other "
            f"channel. Nothing emitted.", where=f"node '{node_name}'")
    value = float(arr.reshape(-1)[0])
    if not (value > 0.0):
        raise Refusal(f"{which} must be a positive constant, got {value}. "
                      f"Nothing emitted.", where=f"node '{node_name}'")
    return value


def _zero_int8_zero_point(graph, name, node_name, which):
    arr = np.asarray(_initializer_or_refuse(graph, name, node_name, which))
    if arr.dtype != np.int8:
        raise Refusal(
            f"{which} is {arr.dtype}; cim.mvm and the simulator are signed "
            f"throughout. Nothing emitted.", where=f"node '{node_name}'")
    if np.any(arr != 0):
        raise Refusal(
            f"{which} is {arr.ravel().tolist()}, but only symmetric "
            f"(zero_point 0) quantization is supported -- a non-zero zero "
            f"point needs a per-output bias term this dialect has no op "
            f"for. Nothing emitted.", where=f"node '{node_name}'")


def _int_attr(node, name, default):
    attr = next((a for a in node.attribute if a.name == name), None)
    if attr is None:
        return default
    return int(attr.i)


def _int_list_attr(node, name, default):
    attr = next((a for a in node.attribute if a.name == name), None)
    if attr is None:
        return list(default)
    return [int(v) for v in attr.ints]


def _str_attr(node, name, default):
    attr = next((a for a in node.attribute if a.name == name), None)
    if attr is None:
        return default
    return attr.s.decode("utf-8")


def load_qlinear_conv(model):
    """The single accepted QLinearConv node.

    Returns (node, weight2d[Cout, Cin*Kh*Kw], x_name, x_shape[N,Cin,H,W],
    conv_params, derived_scale) where `conv_params` is (kh, kw, stride_h,
    stride_w, pad_top, pad_bottom, pad_left, pad_right) -- exactly
    im2col_nchw's own trailing arguments, so callers pass it straight
    through -- and `derived_scale` is the single positive float that makes
    `cim.requantize(scale=derived_scale, zero_point=0, effective_bits=8)`
    compute the same `round(res * (x_scale * w_scale / y_scale))`
    QLinearConv's own reference semantics define, given `res` is the raw
    (pre-scale) accumulator: `round(res / (y_scale / (x_scale * w_scale)))
    == round(res * (x_scale * w_scale / y_scale))`.

    Raises Refusal for anything outside the scope this module's own header
    comment names (grouped/dilated convolution, per-channel quantization,
    a non-zero zero point, a bias operand, non-explicit padding).
    """
    from onnx import numpy_helper

    _check_opset(model)
    graph = model.graph

    convs = [n for n in graph.node if n.op_type == ACCEPTED_CONV_OP]
    if not convs:
        seen = sorted({n.op_type for n in graph.node})
        raise Refusal(
            f"no {ACCEPTED_CONV_OP} node in the graph (found: "
            f"{', '.join(seen) if seen else 'no nodes at all'}). Nothing "
            f"emitted.")
    if len(convs) > 1:
        raise Refusal(
            f"found {len(convs)} {ACCEPTED_CONV_OP} nodes; only a single "
            f"standalone convolution is imported today, not a chain of "
            f"them. Nothing emitted.")

    node = convs[0]
    where = f"node '{node.name or ACCEPTED_CONV_OP}'"

    others = [n for n in graph.node if n is not node]
    if others:
        kinds = sorted({n.op_type for n in others})
        raise Refusal(
            f"graph also contains {', '.join(kinds)}. This front end imports "
            f"a graph consisting of one {ACCEPTED_CONV_OP} and nothing else. "
            f"Nothing emitted.")

    initializers = _tensor_names(model)
    if len(node.input) < 8:
        raise Refusal(
            f"{ACCEPTED_CONV_OP} has {len(node.input)} inputs; x, x_scale, "
            f"x_zero_point, w, w_scale, w_zero_point, y_scale and "
            f"y_zero_point are all mandatory per the ONNX spec. Nothing "
            f"emitted.", where=where)
    (x_name, x_scale_name, x_zp_name, w_name, w_scale_name, w_zp_name,
     y_scale_name, y_zp_name) = node.input[:8]
    if len(node.input) > 8 and node.input[8]:
        raise Refusal(
            f"has a bias operand '{node.input[8]}'. QLinearConv adds it "
            f"directly to the raw int32 accumulator, per output channel, "
            f"before scaling -- there is no cim op that does that at this "
            f"point in the pipeline without risking an interaction with "
            f"cim-partition's own (unrelated) use of cim.reduce_partial "
            f"for K-tiling, which this front end has not verified composes "
            f"correctly. Nothing emitted.", where=where)

    if w_name not in initializers:
        raise Refusal(
            f"the weight operand '{w_name}' is not a constant initializer. "
            f"Weight-stationary hardware has nothing to make resident if "
            f"the weights are computed at runtime. Nothing emitted.",
            where=where)
    if x_name in initializers:
        raise Refusal(
            f"both operands are constant initializers. Nothing emitted.",
            where=where)

    x_scale = _positive_scalar_scale(graph, x_scale_name, node.name, "x_scale")
    w_scale = _positive_scalar_scale(graph, w_scale_name, node.name, "w_scale")
    y_scale = _positive_scalar_scale(graph, y_scale_name, node.name, "y_scale")
    _zero_int8_zero_point(graph, x_zp_name, node.name, "x_zero_point")
    _zero_int8_zero_point(graph, w_zp_name, node.name, "w_zero_point")
    _zero_int8_zero_point(graph, y_zp_name, node.name, "y_zero_point")

    group = _int_attr(node, "group", 1)
    if group != 1:
        raise Refusal(
            f"group={group}; only group=1 (a plain, non-grouped "
            f"convolution) is imported. Each group is really an "
            f"independent matmul over a slice of channels, not one "
            f"reshape. Nothing emitted.", where=where)

    dilations = _int_list_attr(node, "dilations", [1, 1])
    if dilations != [1, 1]:
        raise Refusal(
            f"dilations={dilations}; only dilation (1, 1) is imported. A "
            f"dilated kernel tap reads a non-contiguous patch, which this "
            f"front end's im2col does not express. Nothing emitted.",
            where=where)

    auto_pad = _str_attr(node, "auto_pad", "NOTSET")
    if auto_pad != "NOTSET":
        raise Refusal(
            f"auto_pad='{auto_pad}'; only 'NOTSET' (explicit `pads`, "
            f"defaulting to no padding) is imported. SAME_UPPER/SAME_LOWER/"
            f"VALID all need shape-dependent padding math this front end "
            f"does not replicate -- see this module's own 'WHAT IS "
            f"DELIBERATELY REFUSED' note on why VALID specifically is not "
            f"special-cased despite being definitionally just "
            f"pads=[0, 0, 0, 0] (an onnx.reference bug in exactly that "
            f"computation, not a limitation on this front end's side).  "
            f"Pass pads=[0, 0, 0, 0] explicitly (already this function's "
            f"own default) for the same effect. Nothing emitted.",
            where=where)

    w_init = next(i for i in graph.initializer if i.name == w_name)
    weight = numpy_helper.to_array(w_init)
    if weight.dtype != np.int8:
        raise Refusal(
            f"the weight operand '{w_name}' has dtype {weight.dtype}. "
            f"Nothing emitted.", where=where)
    if weight.ndim != 4:
        raise Refusal(
            f"the weight operand '{w_name}' has rank {weight.ndim}; a 2-D "
            f"convolution's kernel is rank 4 [Cout, Cin, Kh, Kw]. Nothing "
            f"emitted.", where=where)
    cout, cin, kh, kw = weight.shape

    kernel_shape = _int_list_attr(node, "kernel_shape", [kh, kw])
    if kernel_shape != [kh, kw]:
        raise Refusal(
            f"kernel_shape={kernel_shape} does not match the weight's own "
            f"({kh}, {kw}). Nothing emitted.", where=where)

    # auto_pad == "NOTSET" always, checked above.
    pads = _int_list_attr(node, "pads", [0, 0, 0, 0])
    if len(pads) != 4:
        raise Refusal(
            f"pads={pads} has {len(pads)} entries; a 2-D convolution needs "
            f"exactly 4 (pad_top, pad_left, pad_bottom, pad_right). Nothing "
            f"emitted.", where=where)
    pad_top, pad_left, pad_bottom, pad_right = pads

    strides = _int_list_attr(node, "strides", [1, 1])
    if len(strides) != 2:
        raise Refusal(
            f"strides={strides} has {len(strides)} entries; a 2-D "
            f"convolution needs exactly 2. Nothing emitted.", where=where)
    stride_h, stride_w = strides

    x_info = next((v for v in graph.input if v.name == x_name), None)
    if x_info is None:
        raise Refusal(
            f"the activation operand '{x_name}' is neither a graph input "
            f"nor a constant. Nothing emitted.", where=where)
    if x_info.type.tensor_type.elem_type != _INT8:
        raise Refusal(
            f"the activation operand '{x_name}' is not int8. Nothing "
            f"emitted.", where=where)
    x_shape = _shape_of(x_info)
    if x_shape is None:
        raise Refusal(
            f"the activation operand '{x_name}' has a symbolic or unknown "
            f"shape. Nothing emitted.", where=where)
    if len(x_shape) != 4:
        raise Refusal(
            f"the activation operand '{x_name}' has rank {len(x_shape)}; a "
            f"2-D convolution's input is rank 4 [N, Cin, H, W]. Nothing "
            f"emitted.", where=where)
    if x_shape[1] != cin:
        raise Refusal(
            f"activation has Cin={x_shape[1]} but the weight is "
            f"{list(weight.shape)} (Cin={cin}). Nothing emitted.",
            where=where)

    # Cout-major already -- see im2col.py's own note on why, unlike
    # MatMulInteger's B, this weight needs no transpose.
    weight2d = np.ascontiguousarray(weight.reshape(cout, cin * kh * kw))

    # round(res / derived_scale) == round(res * (x_scale * w_scale /
    # y_scale)), QLinearConv's own reference formula for `res` an integer
    # (raw, pre-scale) accumulator -- see this function's own docstring.
    derived_scale = y_scale / (x_scale * w_scale)

    conv_params = (kh, kw, stride_h, stride_w,
                  pad_top, pad_bottom, pad_left, pad_right)
    return node, weight2d, x_name, tuple(x_shape), conv_params, derived_scale


def _validate_conv_activation(activation, x_shape):
    """The supplied activation must be exactly `x_shape` ([N, Cin, H, W])."""
    activation = np.asarray(activation)
    if tuple(activation.shape) != tuple(x_shape):
        raise Refusal(
            f"the supplied activation has shape {list(activation.shape)}; "
            f"the model's convolution input is {list(x_shape)}. Nothing "
            f"emitted.")
    if activation.dtype != np.int8:
        lo, hi = int(activation.min()), int(activation.max())
        if lo < -128 or hi > 127:
            raise Refusal(
                f"the supplied activation has values in [{lo}, {hi}], which "
                f"does not fit int8. Nothing emitted.")
        activation = activation.astype(np.int8)
    return activation


def provenance(model_path, model_bytes, activation_source):
    """MLIR comments recording where this module came from.

    Comments rather than module attributes on purpose: an attribute would
    surface in every cim-opt round trip and in every FileCheck test's
    output, whereas a stray .mlir file on someone's disk is exactly when
    you want to know which model and which activation produced it.
    """
    digest = hashlib.sha256(model_bytes).hexdigest()[:16]
    return (
        f"// Generated by cim_frontend from {model_path}\n"
        f"//   model sha256: {digest}...\n"
        f"//   activation:   {activation_source}\n"
        f"//   NOTE: this module is ONE BAKED INFERENCE -- the activation is a\n"
        f"//   constant, because cim-run takes no runtime inputs.\n")


def _validate_activation(activation, expected_k):
    """Accepts a flat [K] vector (M == 1) or a real [M, K] batch, M >= 1.

    Always RETURNS a 2-D [M, K] array: emit_module/emit_chain_module also
    accept a 1-D [K] array directly (and reshape it the same way, so the
    M == 1 shape stays byte-identical either way), but normalizing here
    means every caller downstream of this function -- both single-layer
    and chain -- sees the same shape regardless of which form was
    supplied.
    """
    activation = np.asarray(activation)
    if activation.ndim == 1:
        activation = activation.reshape(1, -1)
    if activation.ndim != 2:
        raise Refusal(
            f"the supplied activation has shape {list(activation.shape)}; "
            f"expected [K] or [M, K]. Nothing emitted.")
    if activation.shape[1] != expected_k:
        raise Refusal(
            f"the supplied activation has K={activation.shape[1]} but the "
            f"model's K is {expected_k}. Nothing emitted.")
    if activation.dtype != np.int8:
        lo, hi = int(activation.min()), int(activation.max())
        if lo < -128 or hi > 127:
            raise Refusal(
                f"the supplied activation has values in [{lo}, {hi}], which "
                f"does not fit int8. Wrapping them silently would compute with "
                f"different numbers than were supplied. Nothing emitted.")
        activation = activation.astype(np.int8)
    return activation


def import_model(model, activation, model_path="<model>", model_bytes=b"",
                 activation_source="<supplied>"):
    """An ONNX model plus an activation vector -> MLIR text.

    Dispatches on how many MatMulInteger nodes the graph has: exactly one
    goes through the single-layer path (load_matmul_integer, unchanged
    since before chains existed); two or more go through load_matmul_chain.
    A model with zero MatMulInteger nodes but a single QLinearConv goes
    through load_qlinear_conv instead (see that function's own module
    section header for scope and rationale). A model with zero of either,
    or with stray Cast/QuantizeLinear nodes around a single MatMulInteger,
    is refused by load_matmul_integer's own "graph also contains ..."
    check, which already says the right thing for that case.
    """
    from onnx import TensorProto  # noqa: F401 (documents op_type below)

    matmul_count = sum(1 for n in model.graph.node
                       if n.op_type == ACCEPTED_OP)
    conv_count = sum(1 for n in model.graph.node
                     if n.op_type == ACCEPTED_CONV_OP)

    if matmul_count == 0 and conv_count >= 1:
        (node, weight2d, x_name, x_shape, conv_params,
         derived_scale) = load_qlinear_conv(model)
        activation = _validate_conv_activation(activation, x_shape)

        patches, (n_batch, out_h, out_w) = im2col_nchw(activation,
                                                        *conv_params)

        taken = set()
        weight_sym = sanitize_symbol(node.input[3], taken)  # w
        act_sym = sanitize_symbol(x_name, taken)

        header = provenance(model_path, model_bytes, activation_source)
        header += (
            f"//   conv im2col: N={n_batch} OutH={out_h} OutW={out_w} -- "
            f"raw matmul output rows are in (n, oh, ow) row-major order, "
            f"reshape to [N, OutH, OutW, Cout] then transpose(0, 3, 1, 2) "
            f"for ONNX's own [N, Cout, OutH, OutW] layout.\n")
        return emit_module(
            weight2d, patches, weight_sym=weight_sym, act_sym=act_sym,
            header=header,
            trailing_requantize=(derived_scale, 0, 8))

    if matmul_count >= 2:
        nodes, weights, first_act_name, scales = load_matmul_chain(model)
        activation = _validate_activation(activation, weights[0].shape[1])

        taken = set()
        weight_syms = [sanitize_symbol(n.input[1], taken) for n in nodes]
        act_sym = sanitize_symbol(first_act_name, taken)

        header = provenance(model_path, model_bytes, activation_source)
        return emit_chain_module(weights, activation, weight_syms=weight_syms,
                                 act_sym=act_sym, header=header,
                                 scales=scales)

    node, weight, act_name = load_matmul_integer(model)
    activation = _validate_activation(activation, weight.shape[1])

    taken = set()
    weight_sym = sanitize_symbol(node.input[1], taken)
    act_sym = sanitize_symbol(act_name, taken)

    header = provenance(model_path, model_bytes, activation_source)
    return emit_module(weight, activation, weight_sym=weight_sym,
                       act_sym=act_sym, header=header)
