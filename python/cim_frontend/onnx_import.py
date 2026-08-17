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

from .emit import (emit_chain_module, emit_conv_chain_module, emit_module,
                   sanitize_symbol)
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
# A REAL MODEL FORCED A SECOND LOOK: PER-CHANNEL SCALE, BIAS, ASYMMETRIC
# ZERO POINTS
# =========================================================================
# The first version of this function refused all three of the above --
# reasonably, since they looked like real dialect-level gaps. Importing an
# actual quantized model (squeezenet1.0-12-int8 from the ONNX model zoo)
# showed otherwise: EVERY layer of a real, ONNX-Runtime-quantized model
# uses per-channel w_scale, a real int32 bias, and an asymmetric (non-
# zero) x_zero_point -- none of which are exotic, all three are the
# default output of a real quantization toolchain. Investigating each one
# by hand-building a probe module and running it through a real cim-opt/
# cim-run round trip (not by inspection alone) found that all three are
# expressible using ONLY existing, already-verified machinery:
#
#   * Per-channel scale: `cim.requantize` still takes one scalar `scale`,
#     but nothing stops emitting N of them -- one per output channel,
#     each against a `memref.subview` of that column. `AnyMemRef` accepts
#     a strided (non-contiguous) view, and the interpreter's gather/
#     scatter already handle strides generically (the same machinery
#     cim-partition's own ragged-tile padding relies on). See
#     emit.emit_module's `per_channel_requantize` parameter.
#   * Bias: `cim.reduce_partial`'s verifier constrains operand shape and
#     element type only, never producer -- and cim-partition treats a
#     matmul's `outs` buffer as an opaque, pre-allocated destination it
#     fills in place, erasing the matmul op once the tiled sequence has
#     written the real answer back into that SAME buffer. A hand-emitted
#     `cim.reduce_partial(matmul_output, bias_broadcast)`, sitting between
#     the matmul and the eventual requantize, therefore composes with
#     cim-partition's rewrite with no special-casing on either side --
#     confirmed by a real round trip before this parameter existed. See
#     emit.emit_module's `bias` parameter.
#   * A non-zero x_zero_point: this project's own "one baked inference"
#     contract (this file's own module docstring) makes the activation a
#     compile-time constant, so `X - x_zero_point` can be computed once,
#     in Python, before any MLIR is emitted -- exactly like im2col itself.
#     Requires w_zero_point == 0 (see below): QLinearConv's true
#     accumulator is `sum (X - x_zp) * (W - w_zp)`, which only reduces to
#     a plain matmul over `X' = X - x_zp` when the `w_zp` cross term
#     vanishes. Refused, with the exact out-of-range values named, if the
#     shifted activation does not fit signed int8 -- the same "refuse
#     rather than silently wrap" discipline as everywhere else in this
#     front end.
#   * A non-zero y_zero_point: `cim.requantize` already takes a real
#     `zero_point : i32` attribute (`legalize_precision_e2e_test.cpp`
#     proves it functions at zero_point=3), and `cimrt_requantize`'s
#     order of operations -- round, THEN add zero_point, THEN clamp --
#     already matches QLinearConv's own reference formula exactly. The
#     only change needed was to stop hard-coding 0.
#   * dilations != (1, 1): "a dilated kernel tap reads a non-contiguous
#     patch" is true, but non-contiguous refers to the SOURCE indices
#     only -- the destination patch im2col_nchw builds per output
#     position is the same dense [C, Kh, Kw] block either way. numpy's
#     own strided slicing (`start:stop:step`) reads exactly that pattern
#     with the step set to the dilation factor, so this is a sampling
#     change inside the existing loop, not a different data structure --
#     see im2col.py's own "DILATION IS A SAMPLING PATTERN" note. Unlike
#     the four reshapes above, this one needed no real model to motivate
#     it: the shape argument alone was sufficient once looked at closely,
#     which is itself worth naming, since the other four were each found
#     by importing squeezenet1.0-12-int8 rather than by inspection.
#
# What is STILL refused, because it is not one of these reshapes:
#
#   * group != 1: each group is really an independent matmul over a slice
#     of channels, not one reshape. Silently computing group=1's answer
#     for a grouped (e.g. depthwise) conv would be a confident wrong
#     number, not a shape error.
#   * a non-zero w_zero_point (asymmetric WEIGHT quantization): the cross
#     term `w_zp * X` above is per-ROW (activation-dependent), not a fixed
#     per-channel correction the way a bias is -- genuinely not a reshape
#     of what already exists. In practice this is also the least costly
#     restriction to keep: real quantization toolchains commonly keep
#     weights symmetric (w_zero_point == 0) even when activations are
#     asymmetric, exactly as squeezenet1.0-12-int8 itself does.
#   * a non-scalar x_zero_point or y_zero_point: per-tensor activation/
#     output quantization is the near-universal convention (unlike
#     per-channel WEIGHT scale, which is common); a per-channel x/y zero
#     point would need the same N-way subview treatment as `w_scale`
#     above, not yet implemented.
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
_UINT8 = 2  # TensorProto.UINT8


def _initializer_or_refuse(graph, name, node_name, which):
    arr = _initializer_array(graph, name)
    if arr is None:
        raise Refusal(
            f"{which} '{name}' is not a constant initializer, so its value "
            f"cannot be checked at compile time. Nothing emitted.",
            where=f"node '{node_name}'")
    return arr


def _positive_scalar_scale(graph, name, node_name, which):
    """A scale that must be exactly one value -- x_scale and y_scale,
    which are per-tensor (never per-channel) in every quantization
    convention this front end has seen."""
    arr = np.asarray(_initializer_or_refuse(graph, name, node_name, which))
    if arr.size != 1:
        raise Refusal(
            f"{which} has {arr.size} elements; only per-tensor (scalar) "
            f"quantization is supported here. Nothing emitted.",
            where=f"node '{node_name}'")
    value = float(arr.reshape(-1)[0])
    if not (value > 0.0):
        raise Refusal(f"{which} must be a positive constant, got {value}. "
                      f"Nothing emitted.", where=f"node '{node_name}'")
    return value


def _positive_weight_scale(graph, name, node_name, which, cout):
    """w_scale: a single scalar (applies to every output channel), OR
    exactly `cout` values (one per output channel, real per-channel
    quantization -- see this module's own 'A REAL MODEL FORCED A SECOND
    LOOK' note). Returns a Python float for the scalar case, or a numpy
    array of length `cout` for the per-channel case -- the caller tells
    the two apart with `np.isscalar`/`isinstance(..., float)` vs ndarray,
    mirroring emit_module's own trailing_requantize/per_channel_requantize
    split.
    """
    arr = np.asarray(_initializer_or_refuse(graph, name, node_name, which))
    if arr.size not in (1, cout):
        raise Refusal(
            f"{which} has {arr.size} elements; expected either 1 (a single "
            f"scale for every output channel) or {cout} (one per output "
            f"channel, this node's Cout). Nothing emitted.",
            where=f"node '{node_name}'")
    flat = arr.reshape(-1).astype(np.float64)
    bad = flat[~(flat > 0.0)]
    if bad.size:
        raise Refusal(
            f"{which} must be all-positive constants, got {flat.tolist()}. "
            f"Nothing emitted.", where=f"node '{node_name}'")
    return float(flat[0]) if arr.size == 1 else flat


def _scalar_zero_point(graph, name, node_name, which, allow_nonzero):
    """A zero point that must be exactly one value (per-tensor, never
    per-channel -- see load_qlinear_conv's own scope note). Returns
    (value: int, dtype: np.dtype). Does not itself enforce zero; callers
    that must stay symmetric (w_zero_point) pass allow_nonzero=False.
    """
    arr = np.asarray(_initializer_or_refuse(graph, name, node_name, which))
    if arr.dtype not in (np.int8, np.uint8):
        raise Refusal(
            f"{which} is {arr.dtype}; expected int8 or uint8. Nothing "
            f"emitted.", where=f"node '{node_name}'")
    if arr.size != 1:
        raise Refusal(
            f"{which} has {arr.size} elements; only per-tensor (scalar) "
            f"zero points are supported here. Nothing emitted.",
            where=f"node '{node_name}'")
    value = int(arr.reshape(-1)[0])
    if not allow_nonzero and value != 0:
        raise Refusal(
            f"{which} is {value}, but only symmetric (zero_point 0) "
            f"quantization is supported here. Nothing emitted.",
            where=f"node '{node_name}'")
    return value, arr.dtype


def _weight_zero_point_must_be_zero(graph, name, node_name, cout):
    """w_zero_point: a single scalar, OR (like w_scale) `cout` values, one
    per output channel -- real quantization tools commonly emit a
    per-channel array here even when, as in every model this front end
    has been tested against, every entry is trivially 0 (a real
    squeezenet1.0-12-int8 layer does exactly this: a length-64
    w_zero_point where all 64 entries are 0). Either shape is accepted,
    but every element must be 0 either way -- this is still the one zero
    point load_qlinear_conv never accepts non-zero (see its own module
    section header for why).
    """
    which = "w_zero_point"
    arr = np.asarray(_initializer_or_refuse(graph, name, node_name, which))
    if arr.dtype != np.int8:
        raise Refusal(
            f"{which} is {arr.dtype}; expected int8. Nothing emitted.",
            where=f"node '{node_name}'")
    if arr.size not in (1, cout):
        raise Refusal(
            f"{which} has {arr.size} elements; expected either 1 or "
            f"{cout} (one per output channel, matching w_scale's own "
            f"shape). Nothing emitted.", where=f"node '{node_name}'")
    if np.any(arr != 0):
        raise Refusal(
            f"{which} is {arr.ravel().tolist()}, but only symmetric "
            f"(zero_point 0) weight quantization is supported. Nothing "
            f"emitted.", where=f"node '{node_name}'")


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


def _conv_geometry(node, weight, where):
    """group/dilations/auto_pad/kernel_shape/pads/strides -- validated the
    same way for every QLinearConv node this front end reads, whether
    standalone (load_qlinear_conv) or the first layer of a chain
    (load_conv_matmul_chain). Factored out rather than duplicated: two
    copies of the same validation drifting apart is exactly the failure
    class analyze.py's own uint8-y_zero_point regression was (see
    docs/roadmap.md's Phase 1 hardening entry) -- one copy, called from
    both places, cannot go stale relative to itself.

    Returns (dilation_h, dilation_w, pad_top, pad_bottom, pad_left,
    pad_right, stride_h, stride_w).
    """
    cout, cin, kh, kw = weight.shape

    group = _int_attr(node, "group", 1)
    if group != 1:
        raise Refusal(
            f"group={group}; only group=1 (a plain, non-grouped "
            f"convolution) is imported. Each group is really an "
            f"independent matmul over a slice of channels, not one "
            f"reshape. Nothing emitted.", where=where)

    dilations = _int_list_attr(node, "dilations", [1, 1])
    if len(dilations) != 2:
        raise Refusal(
            f"dilations={dilations} has {len(dilations)} entries; a 2-D "
            f"convolution needs exactly 2. Nothing emitted.", where=where)
    dilation_h, dilation_w = dilations
    if dilation_h <= 0 or dilation_w <= 0:
        raise Refusal(
            f"dilations={dilations}; both entries must be positive. "
            f"Nothing emitted.", where=where)

    auto_pad = _str_attr(node, "auto_pad", "NOTSET")
    if auto_pad != "NOTSET":
        raise Refusal(
            f"auto_pad='{auto_pad}'; only 'NOTSET' (explicit `pads`, "
            f"defaulting to no padding) is imported. SAME_UPPER/SAME_LOWER/"
            f"VALID all need shape-dependent padding math this front end "
            f"does not replicate -- see load_qlinear_conv's own 'WHAT IS "
            f"DELIBERATELY REFUSED' note on why VALID specifically is not "
            f"special-cased despite being definitionally just "
            f"pads=[0, 0, 0, 0] (an onnx.reference bug in exactly that "
            f"computation, not a limitation on this front end's side).  "
            f"Pass pads=[0, 0, 0, 0] explicitly (already this function's "
            f"own default) for the same effect. Nothing emitted.",
            where=where)

    kernel_shape = _int_list_attr(node, "kernel_shape", [kh, kw])
    if kernel_shape != [kh, kw]:
        raise Refusal(
            f"kernel_shape={kernel_shape} does not match the weight's own "
            f"({kh}, {kw}). Nothing emitted.", where=where)

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

    return (dilation_h, dilation_w, pad_top, pad_bottom, pad_left,
           pad_right, stride_h, stride_w)


def load_qlinear_conv(model):
    """The single accepted QLinearConv node.

    Returns (node, weight2d[Cout, Cin*Kh*Kw], x_name, x_shape[N,Cin,H,W],
    conv_params, x_zero_point, trailing_requantize, per_channel_requantize,
    bias, uint8_output_shifted) where:

    - `conv_params` is (kh, kw, stride_h, stride_w, pad_top, pad_bottom,
      pad_left, pad_right, dilation_h, dilation_w) -- exactly
      im2col_nchw's own trailing arguments, so callers pass it straight
      through.
    - `x_zero_point` is a plain int; the caller subtracts it from the
      supplied activation BEFORE im2col (see this module's own 'A REAL
      MODEL FORCED A SECOND LOOK' note above for why that is exact when
      w_zero_point == 0, which is always true here).
    - Exactly one of `trailing_requantize` (a `(scale, zero_point,
      effective_bits)` triple, for a scalar w_scale) or
      `per_channel_requantize` (a length-Cout list of `(scale,
      zero_point)` pairs, for a per-channel w_scale) is not None --
      callers pass whichever straight through to emit.emit_module. Both
      forms derive their scale(s) from `y_scale / (x_scale * w_scale)`,
      matching QLinearConv's own reference formula
      `round(res * (x_scale * w_scale / y_scale))` for `res` an integer
      (raw, pre-scale) accumulator. The `zero_point` they carry is
      ALREADY the emitted-module's own zero point, which is
      `y_zero_point` unchanged for an int8 output but `y_zero_point -
      128` for a uint8 one -- see `uint8_output_shifted` below.
    - `bias` is None, or a length-Cout int32 numpy array -- the caller
      passes it straight through to emit.emit_module's own `bias`
      parameter.
    - `uint8_output_shifted` is a bool: True when the ONNX node declares
      a uint8 `y_zero_point` (and so a uint8 output). `cim.requantize`'s
      clamp is a signed `effective_bits` range, so the emitted module's
      result is exactly `(true_uint8_output - 128)` in every element,
      not the true value -- callers MUST disclose this in the module's
      provenance header (import_model does). False for the ordinary
      int8-output case, where the emitted result needs no adjustment.

    Raises Refusal for anything outside the scope this module's own
    header comment names (grouped/dilated convolution, a non-zero
    w_zero_point, a per-channel x/y zero point, non-explicit padding).
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
    bias_name = node.input[8] if len(node.input) > 8 else ""

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

    # Read the weight early: Cout has to be known before w_scale (which
    # may be per-channel) or a bias (which is always per-channel) can be
    # validated.
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

    x_scale = _positive_scalar_scale(graph, x_scale_name, node.name, "x_scale")
    w_scale = _positive_weight_scale(graph, w_scale_name, node.name,
                                     "w_scale", cout)
    y_scale = _positive_scalar_scale(graph, y_scale_name, node.name, "y_scale")
    x_zero_point, _x_zp_dtype = _scalar_zero_point(
        graph, x_zp_name, node.name, "x_zero_point", allow_nonzero=True)
    _weight_zero_point_must_be_zero(graph, w_zp_name, node.name, cout)
    y_zero_point, y_zp_dtype = _scalar_zero_point(
        graph, y_zp_name, node.name, "y_zero_point", allow_nonzero=True)
    # cim.requantize's clamp is a SIGNED effective_bits range (see
    # cimrt_requantize), which cannot represent a uint8 output's full
    # [0, 255] span directly -- but it does not need to. For any real t,
    # clamp(-128, 127, t - 128) == clamp(0, 255, t) - 128: both clamp
    # bounds shift by exactly 128, so the affine shift commutes with
    # clamping exactly, not approximately. Requantizing with
    # (y_zero_point - 128) instead of y_zero_point therefore produces
    # EXACTLY (true_uint8_output - 128) in every element, never a lucky
    # coincidence at the extremes. `uint8_output_shifted` tells the
    # caller (import_model) to disclose that shift in the EMITTED
    # MODULE's own provenance header, on the same "the front end's job
    # ends at 'produce the right numbers'; a documented caller-side
    # transform is not a silent one" discipline as the NHWC reshape a
    # conv's own caller already has to apply (import_model's own
    # "conv im2col: ..." header line).
    uint8_output_shifted = y_zp_dtype == np.uint8
    if uint8_output_shifted:
        y_zero_point -= 128

    bias = None
    if bias_name:
        bias = np.asarray(_initializer_or_refuse(
            graph, bias_name, node.name, "B")).astype(np.int64)
        if bias.shape != (cout,):
            raise Refusal(
                f"bias 'B' has shape {list(bias.shape)}; expected a single "
                f"length-Cout ({cout}) int32 array, one value per output "
                f"channel. Nothing emitted.", where=where)
        bias = bias.astype(np.int32)

    (dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right,
     stride_h, stride_w) = _conv_geometry(node, weight, where)

    x_info = next((v for v in graph.input if v.name == x_name), None)
    if x_info is None:
        raise Refusal(
            f"the activation operand '{x_name}' is neither a graph input "
            f"nor a constant. Nothing emitted.", where=where)
    if x_info.type.tensor_type.elem_type not in (_INT8, _UINT8):
        raise Refusal(
            f"the activation operand '{x_name}' is not int8 or uint8. "
            f"Nothing emitted.", where=where)
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
    # w_scale is either a single float (uniform) or a length-Cout array
    # (per-channel); the two cases produce trailing_requantize XOR
    # per_channel_requantize, matching emit.emit_module's own split.
    if isinstance(w_scale, np.ndarray):
        derived_scales = y_scale / (x_scale * w_scale)
        trailing_requantize = None
        per_channel_requantize = [(float(s), y_zero_point)
                                  for s in derived_scales]
    else:
        derived_scale = y_scale / (x_scale * w_scale)
        trailing_requantize = (derived_scale, y_zero_point, 8)
        per_channel_requantize = None

    conv_params = (kh, kw, stride_h, stride_w,
                  pad_top, pad_bottom, pad_left, pad_right,
                  dilation_h, dilation_w)
    return (node, weight2d, x_name, tuple(x_shape), conv_params,
           x_zero_point, trailing_requantize, per_channel_requantize, bias,
           uint8_output_shifted)


# ---------------------------------------------------------------------------
# A QLinearConv feeding one or more MatMulInteger layers -- the first real
# multi-layer CNN capability step (docs/roadmap.md's "chain convolution
# layers" plan).
#
# WHY THIS IS TRACTABLE WITHOUT AN MLIR-LEVEL IM2COL
# ====================================================
# Chaining a SECOND convolution needs im2col to run again against a value
# that is only known once the first layer's MLIR actually executes -- no
# longer a concrete Python array. That is a real, separate piece of work
# (a genuine MLIR-level im2col; the interpreter cannot even execute the
# index arithmetic a naive loop-based one would need -- see docs/roadmap.md).
# Chaining a QLinearConv into a MatMulInteger needs none of that: the CONV
# is always layer 0, so its activation is still the literal graph input,
# im2col still runs in Python exactly once, exactly as load_qlinear_conv's
# own docstring already establishes, and everything after it is the
# already-proven MatMulInteger chain bridge (emit_chain_module). This
# function's whole job is producing the (weight2d, patches) pair
# load_qlinear_conv already computes for layer 0, plus the matmul layers'
# already-transposed weights, in exactly the shape emit_chain_module
# already accepts -- no new emission code, matching how load_qlinear_conv's
# own single-conv case reused emit_module unchanged.
#
# THE CONV->MATMUL BRIDGE: Transpose(perm=[0,2,3,1]) -> Reshape([M, Cout])
# ============================================================================
# QLinearConv quantizes its OWN output in-node (via y_scale/y_zero_point),
# unlike two MatMulInteger layers, whose bridge is an explicit
# Cast(to=float32) -> QuantizeLinear pair (see _validate_bridge) -- so there
# is no requantization node to walk here. But the conv's raw ONNX output is
# STILL not directly usable as MatMulInteger's "A" operand: it is
# [N, Cout, OutH, OutW], and ONNX's own MatMulInteger reference kernel does
# real N-D numpy-broadcast matmul rather than requiring 2-D operands --
# confirmed by hand: feeding a 4-D conv output straight into MatMulInteger
# does not error, it silently contracts whatever axes happen to align
# (e.g. OutW against Cout, if they happen to be equal), which is a REAL
# wrong answer, not a shape error, for exactly the class of bug this
# project refuses to ship. So an explicit `Transpose(perm=[0, 2, 3, 1])`
# (NCHW -> NHWC) followed by `Reshape([N*OutH*OutW, Cout])` is REQUIRED in
# the graph between the conv and the first matmul -- not merely tolerated
# -- to make the ONNX graph itself a faithful, standards-compliant
# statement of the same function this compiler emits (the (n, oh, ow)-
# row-major x Cout flatten im2col.py's own patches layout and
# emit_chain_module's per-layer bridge already commit to). Confirmed by
# hand against onnx.reference before this loader was written: the full
# QLinearConv -> Transpose -> Reshape -> MatMulInteger graph's own
# evaluation matches conv's raw NCHW output manually transposed/reshaped
# and matrix-multiplied, exactly.
#
# WHY THIS IS NARROWER THAN A STANDALONE CONV OR A STANDALONE CHAIN
# =====================================================================
# emit_chain_module's bridge is ONE unconditional scalar cim.requantize
# with zero_point hardcoded at 0 (see its own docstring) -- the same
# restriction load_matmul_chain's own bridge already carries via
# _validate_bridge. A standalone QLinearConv can absorb a non-zero
# y_zero_point, a per-channel w_scale, and a bias because load_qlinear_conv
# threads those into emit_module's own trailing_requantize/
# per_channel_requantize/bias parameters -- none of which emit_chain_module
# has an equivalent for. So a conv chained into further layers requires
# y_zero_point == 0, a single scalar w_scale, and no bias. Lifting any of
# these needs emit_chain_module itself to grow a per-channel/bias-aware
# bridge, not just this loader -- deliberately not attempted here, on the
# same "don't invent capability a real model hasn't forced" discipline
# load_qlinear_conv's own module section header describes.
# ---------------------------------------------------------------------------


def load_conv_matmul_chain(model):
    """One QLinearConv (layer 0) followed by one or more MatMulInteger
    layers, bridged as described above.

    Returns (conv_node, matmul_nodes, weight2d, x_name, x_shape,
    conv_params, x_zero_point, matmul_weights, bridge_scales) where:

    - `weight2d`/`x_name`/`x_shape`/`conv_params`/`x_zero_point` are
      exactly load_qlinear_conv's own returns for the conv layer -- the
      caller applies the SAME `_validate_conv_activation` +
      `im2col_nchw(activation, *conv_params)` sequence it would for a
      standalone conv.
    - `matmul_weights[i]` is matmul layer i's ALREADY-TRANSPOSED [N, K]
      weight (same convention as load_matmul_chain's own `weights`).
    - `bridge_scales` has `len(matmul_weights)` entries: `bridge_scales[0]`
      is the conv -> first-matmul bridge's scale (derived from
      `y_scale / (x_scale * w_scale)`, QLinearConv's own reference
      formula, exactly as a standalone conv's `trailing_requantize` scale
      is); `bridge_scales[1:]` are the real QuantizeLinear scales between
      matmul layers, exactly as load_matmul_chain's own `scales` returns.
      Pass `[weight2d] + matmul_weights`, the im2col'd patches matrix, and
      `bridge_scales` straight through to emit.emit_chain_module.

    Raises Refusal for anything outside this function's own scope (see
    this module's section header above): more than one QLinearConv, a
    non-zero conv y_zero_point, a per-channel conv w_scale, a conv bias,
    or anything load_qlinear_conv/load_matmul_chain would themselves
    refuse on their own respective layers.
    """
    from onnx import numpy_helper

    _check_opset(model)
    graph = model.graph
    initializers = _tensor_names(model)

    convs = [n for n in graph.node if n.op_type == ACCEPTED_CONV_OP]
    matmul_nodes = [n for n in graph.node if n.op_type == ACCEPTED_OP]
    if len(convs) != 1:
        raise Refusal(
            f"found {len(convs)} {ACCEPTED_CONV_OP} nodes; "
            f"load_conv_matmul_chain handles exactly one, as the first "
            f"layer of a chain. Nothing emitted.")
    if not matmul_nodes:
        raise Refusal(
            f"no {ACCEPTED_OP} node in the graph; load_conv_matmul_chain "
            f"needs at least one, to chain the convolution into. Use "
            f"load_qlinear_conv for a standalone convolution instead. "
            f"Nothing emitted.")

    allowed = {ACCEPTED_CONV_OP, ACCEPTED_OP, "Cast", "QuantizeLinear",
              "Transpose", "Reshape"}
    others = [n for n in graph.node if n.op_type not in allowed]
    if others:
        kinds = sorted({n.op_type for n in others})
        raise Refusal(
            f"graph also contains {', '.join(kinds)}. A "
            f"{ACCEPTED_CONV_OP} chained into {ACCEPTED_OP} layers may "
            f"only be bridged by Transpose(perm=[0, 2, 3, 1]) -> "
            f"Reshape([M, Cout]) between the conv and the first matmul "
            f"(see this section's own module header for why that exact "
            f"pattern is required, not merely tolerated), and Cast "
            f"(to=float32) -> QuantizeLinear between matmul layers. "
            f"Nothing emitted.")

    conv_node = convs[0]
    where = f"node '{conv_node.name or ACCEPTED_CONV_OP}'"

    if len(conv_node.input) < 8:
        raise Refusal(
            f"{ACCEPTED_CONV_OP} has {len(conv_node.input)} inputs; x, "
            f"x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale and "
            f"y_zero_point are all mandatory per the ONNX spec. Nothing "
            f"emitted.", where=where)
    (x_name, x_scale_name, x_zp_name, w_name, w_scale_name, w_zp_name,
     y_scale_name, y_zp_name) = conv_node.input[:8]
    bias_name = conv_node.input[8] if len(conv_node.input) > 8 else ""

    if bias_name:
        raise Refusal(
            f"this node has a bias operand '{bias_name}', but "
            f"load_conv_matmul_chain does not support a bias on a "
            f"convolution chained into further layers -- "
            f"emit_chain_module's bridge has no cim.reduce_partial step. "
            f"Compile this convolution standalone instead, or drop the "
            f"bias. Nothing emitted.", where=where)
    if w_name not in initializers:
        raise Refusal(
            f"the weight operand '{w_name}' is not a constant initializer. "
            f"Nothing emitted.", where=where)
    if x_name in initializers:
        raise Refusal(
            f"both operands are constant initializers. Nothing emitted.",
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

    x_scale = _positive_scalar_scale(graph, x_scale_name, conv_node.name,
                                     "x_scale")
    w_scale = _positive_weight_scale(graph, w_scale_name, conv_node.name,
                                     "w_scale", cout)
    if isinstance(w_scale, np.ndarray):
        raise Refusal(
            f"w_scale has {w_scale.size} elements; a convolution chained "
            f"into further layers requires a single scalar w_scale, not a "
            f"per-channel one -- emit_chain_module's bridge is one "
            f"unconditional cim.requantize, not the N-way per-channel "
            f"split a standalone conv's own per_channel_requantize uses. "
            f"Compile this convolution standalone instead. Nothing "
            f"emitted.", where=where)
    y_scale = _positive_scalar_scale(graph, y_scale_name, conv_node.name,
                                     "y_scale")
    x_zero_point, _x_zp_dtype = _scalar_zero_point(
        graph, x_zp_name, conv_node.name, "x_zero_point", allow_nonzero=True)
    _weight_zero_point_must_be_zero(graph, w_zp_name, conv_node.name, cout)
    # allow_nonzero=False: _scalar_zero_point itself refuses a non-zero
    # value, with the exact reason emit_chain_module's bridge needs it
    # (see this section's own module header).
    y_zero_point, y_zp_dtype = _scalar_zero_point(
        graph, y_zp_name, conv_node.name, "y_zero_point", allow_nonzero=False)
    if y_zp_dtype != np.int8:
        raise Refusal(
            f"y_zero_point is {y_zp_dtype}; a convolution chained into "
            f"further layers must declare an int8 output. The standalone "
            f"conv path can accept uint8 (load_qlinear_conv's own "
            f"uint8_output_shifted note) because it documents a caller-side "
            f"+128 shift on the FINAL printed result -- but there is no "
            f"caller here to apply that shift before the next matmul reads "
            f"this value, so a uint8-declared output would be silently "
            f"misinterpreted downstream. Nothing emitted.", where=where)

    (dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right,
     stride_h, stride_w) = _conv_geometry(conv_node, weight, where)

    x_info = next((v for v in graph.input if v.name == x_name), None)
    if x_info is None:
        raise Refusal(
            f"the activation operand '{x_name}' is neither a graph input "
            f"nor a constant. Nothing emitted.", where=where)
    if x_info.type.tensor_type.elem_type not in (_INT8, _UINT8):
        raise Refusal(
            f"the activation operand '{x_name}' is not int8 or uint8. "
            f"Nothing emitted.", where=where)
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

    # Cout-major already -- see im2col.py's own note on why a conv kernel
    # needs no transpose the way MatMulInteger's B does.
    weight2d = np.ascontiguousarray(weight.reshape(cout, cin * kh * kw))
    conv_scale = y_scale / (x_scale * w_scale)

    # Output spatial size -- same formula as im2col_nchw's own (see that
    # function's docstring); needed here only to validate the Reshape
    # bridge's target shape below, not to gather any patches (im2col
    # itself runs once the caller has a real activation array).
    n_batch = x_shape[0]
    dilated_kh = (kh - 1) * dilation_h + 1
    dilated_kw = (kw - 1) * dilation_w + 1
    hp = x_shape[2] + pad_top + pad_bottom
    wp = x_shape[3] + pad_left + pad_right
    out_h = (hp - dilated_kh) // stride_h + 1
    out_w = (wp - dilated_kw) // stride_w + 1
    m = n_batch * out_h * out_w

    # The conv -> first-matmul bridge: Transpose(perm=[0, 2, 3, 1]) ->
    # Reshape([M, Cout]) -- see this section's own module header for why
    # this exact pattern is required, not merely tolerated.
    producer = {out: n for n in graph.node for out in n.output}
    consumers = {}
    for n in graph.node:
        for inp in n.input:
            if inp:
                consumers.setdefault(inp, []).append(n)

    conv_out = conv_node.output[0]
    conv_readers = consumers.get(conv_out, [])
    if len(conv_readers) != 1 or conv_readers[0].op_type != "Transpose":
        raise Refusal(
            f"the convolution's output is not read by exactly one "
            f"Transpose node ({len(conv_readers)} reader(s) found). "
            f"Nothing emitted.", where=where)
    transpose_node = conv_readers[0]
    perm_attr = next((a for a in transpose_node.attribute
                      if a.name == "perm"), None)
    perm = list(perm_attr.ints) if perm_attr is not None else None
    if perm != [0, 2, 3, 1]:
        raise Refusal(
            f"Transpose '{transpose_node.name}' has perm={perm}; the "
            f"conv->matmul bridge requires perm=[0, 2, 3, 1] (NCHW -> "
            f"NHWC), matching im2col.py's own (n, oh, ow)-row-major x "
            f"Cout flatten order. Nothing emitted.", where=where)

    transpose_out = transpose_node.output[0]
    transpose_readers = consumers.get(transpose_out, [])
    if len(transpose_readers) != 1 or transpose_readers[0].op_type != "Reshape":
        raise Refusal(
            f"Transpose '{transpose_node.name}''s output is not read by "
            f"exactly one Reshape node ({len(transpose_readers)} "
            f"reader(s) found). Nothing emitted.", where=where)
    reshape_node = transpose_readers[0]
    if len(reshape_node.input) < 2:
        raise Refusal(
            f"Reshape '{reshape_node.name}' has no shape operand. "
            f"Nothing emitted.", where=where)
    shape_arr = _initializer_array(graph, reshape_node.input[1])
    if shape_arr is None:
        raise Refusal(
            f"Reshape '{reshape_node.name}''s shape operand "
            f"'{reshape_node.input[1]}' is not a constant initializer. "
            f"Nothing emitted.", where=where)
    target_shape = [int(v) for v in np.asarray(shape_arr).reshape(-1)]
    if target_shape != [m, cout]:
        raise Refusal(
            f"Reshape '{reshape_node.name}' targets {target_shape}; the "
            f"conv->matmul bridge requires exactly [{m}, {cout}] "
            f"(M = N*OutH*OutW = {m}, Cout = {cout}). A wildcard (-1 or "
            f"0) is not accepted -- this bridge must be an explicit, "
            f"checkable statement of the same shape im2col_nchw derives, "
            f"not something this loader has to infer. Nothing emitted.",
            where=where)

    reshape_out = reshape_node.output[0]
    reshape_readers = consumers.get(reshape_out, [])
    if len(reshape_readers) != 1 or reshape_readers[0] is not matmul_nodes[0]:
        raise Refusal(
            f"Reshape '{reshape_node.name}''s output is not read by "
            f"exactly the first {ACCEPTED_OP} layer "
            f"({len(reshape_readers)} reader(s) found). Nothing "
            f"emitted.", where=where)
    if matmul_nodes[0].input[0] != reshape_out:
        raise Refusal(
            f"the first {ACCEPTED_OP} layer's activation operand is not "
            f"Reshape '{reshape_node.name}''s own output. Nothing "
            f"emitted.", where=where)

    if len(graph.output) != 1:
        raise Refusal(
            f"graph has {len(graph.output)} outputs; a chain must end in "
            f"exactly one. Nothing emitted.")
    if graph.output[0].name != matmul_nodes[-1].output[0]:
        raise Refusal(
            f"the graph's output is not the last {ACCEPTED_OP}'s own "
            f"output -- only a chain that ends directly at the last "
            f"layer's int32 accumulator, with no trailing op, is "
            f"supported. Nothing emitted.")

    matmul_weights = []
    bridge_scales = [conv_scale]
    prev_n = cout
    for i, node in enumerate(matmul_nodes):
        where_i = f"node '{node.name or ACCEPTED_OP}' (matmul layer {i})"
        a_name, b_name = node.input[0], node.input[1]

        if b_name not in initializers:
            raise Refusal(
                f"the weight operand '{b_name}' is not a constant "
                f"initializer. Nothing emitted.", where=where_i)
        if a_name in initializers:
            raise Refusal(
                f"both operands are constant initializers. Nothing "
                f"emitted.", where=where_i)

        a_zp = node.input[2] if len(node.input) > 2 else ""
        b_zp = node.input[3] if len(node.input) > 3 else ""
        _zero_point_is_zero(model, a_zp, node.name or ACCEPTED_OP,
                            "a_zero_point")
        _zero_point_is_zero(model, b_zp, node.name or ACCEPTED_OP,
                            "b_zero_point")

        mm_weight = numpy_helper.to_array(
            next(t for t in graph.initializer if t.name == b_name))
        if mm_weight.dtype != np.int8:
            raise Refusal(
                f"the weight operand '{b_name}' has dtype {mm_weight.dtype}; "
                f"cim.mvm and the simulator are signed throughout. Nothing "
                f"emitted.", where=where_i)
        if mm_weight.ndim != 2:
            raise Refusal(
                f"the weight operand '{b_name}' has rank {mm_weight.ndim}; "
                f"cim-partition requires rank-2 operands. Nothing "
                f"emitted.", where=where_i)
        if mm_weight.shape[0] != prev_n:
            raise Refusal(
                f"layer {i}'s K ({mm_weight.shape[0]}) does not match the "
                f"previous layer's N ({prev_n}). Nothing emitted.",
                where=where_i)

        if i > 0:
            bridge_scales.append(_validate_bridge(
                graph, producer, consumers, matmul_nodes[i - 1].output[0],
                a_name, where_i))

        matmul_weights.append(np.ascontiguousarray(mm_weight.T))
        prev_n = mm_weight.shape[1]

    conv_params = (kh, kw, stride_h, stride_w,
                  pad_top, pad_bottom, pad_left, pad_right,
                  dilation_h, dilation_w)
    return (conv_node, matmul_nodes, weight2d, x_name, tuple(x_shape),
           conv_params, x_zero_point, matmul_weights, bridge_scales)


# ---------------------------------------------------------------------------
# A chain of two or more QLinearConv nodes, connected DIRECTLY -- the real,
# hard capability docs/roadmap.md's "chain convolution layers" plan set out
# to reach (PR C). Depends on emit.emit_conv_chain_module (a real MLIR-level
# im2col for layer 1 onward, via expand_shape/collapse_shape and
# Kh*Kw-unrolled static-stride subview taps -- see that function's own
# docstring for the full design and why it looks the way it does).
#
# WHY THIS NEEDS NO BRIDGE NODE, UNLIKE EVERY OTHER CHAIN THIS FRONT END
# ACCEPTS
# ============================================================================
# load_matmul_chain's bridge is an explicit Cast(to=float32) ->
# QuantizeLinear pair (MatMulInteger's "Y" has no native quantization of its
# own). load_conv_matmul_chain's bridge is an explicit
# Transpose(perm=[0,2,3,1]) -> Reshape([M,Cout]) pair (MatMulInteger has no
# native NCHW/NHWC convention at all, so the conv's own [N,Cout,H,W] output
# needs an explicit statement of how it becomes a 2-D matmul operand).
# QLinearConv needs neither: both its own "X" input and "Y" output are
# ALREADY declared [N,C,H,W] per the ONNX spec, so wiring one conv's "Y"
# straight into the next conv's own "X" is already a faithful,
# standards-compliant statement of the same function this compiler emits --
# there is nothing to bridge.
#
# WHY THIS IS NARROWER THAN A STANDALONE CONV, FOR EVERY LAYER BUT THE LAST
# =============================================================================
# Exactly load_conv_matmul_chain's own restriction, and for the same
# reason: emit_conv_chain_module's own per-bridge cim.requantize is ONE
# unconditional scalar call at zero_point 0 (reused, unchanged, from
# emit_chain_module) -- there is no MLIR-level per-channel-requantize or
# bias-add op available for an intermediate bridge. So every layer except
# the LAST must have y_zero_point == 0, a scalar (non-per-channel) w_scale,
# and no bias. Every layer except the FIRST must ALSO have x_zero_point ==
# 0: its activation arrives already zero-centered, produced by the
# previous layer's own zero_point-0 bridge, and the emitted IR applies no
# zero-point shift to an intermediate activation the way load_qlinear_conv
# applies one to layer 0's (in Python, via _validate_conv_activation,
# before im2col even runs) -- a nonzero x_zero_point here would silently
# diverge from what onnx.reference actually computes for this graph. Only
# the LAST layer keeps load_qlinear_conv's own full generality: asymmetric
# zero point (including uint8, with the exact same -128 shift and
# `uint8_output_shifted` disclosure load_qlinear_conv's own docstring
# describes), a per-channel w_scale, and a bias.
#
# BRIDGE SCALE INDEPENDENCE
# ==========================
# bridge_scales[i] is layer i's OWN y_scale / (x_scale * w_scale) --
# QLinearConv's own reference formula for layer i's raw accumulator,
# exactly load_conv_matmul_chain's own conv_scale. There is no need to
# cross-validate layer i's y_scale against layer i+1's x_scale: QLinearConv's
# reference formula computes a scale-independent RAW integer accumulator
# and only applies scale in each conv's own final step, so each layer is
# self-contained -- layer i+1's own x_scale is used only for layer i+1's
# OWN eventual bridge/output scale, never to reinterpret the already-bridged
# integer values themselves (whose correctness depends only on
# x_zero_point == 0, already required above).
# ---------------------------------------------------------------------------


def load_conv_chain(model):
    """Two or more chained QLinearConv nodes, discovered via producer/
    consumer edges (NOT node-list order -- ONNX does not guarantee a
    topological, let alone chain, node ordering) and connected directly,
    conv_i's own "Y" feeding conv_(i+1)'s own "X" with no bridge node
    between them (see this section's own module header for why).

    Returns (conv_nodes, weights2d, x_name, x_shape, conv_params,
    x_zero_point, trailing_requantize, per_channel_requantize, bias,
    uint8_output_shifted, bridge_scales) where:

    - `conv_nodes` is the chain's own QLinearConv nodes, in execution
      order.
    - `weights2d[0]` is layer 0's weight, Cin-major [Cout, Cin*Kh*Kw] --
      load_qlinear_conv's own convention, no transpose, since layer 0's
      activation is still the literal NCHW graph input. `weights2d[i]`
      for i >= 1 is CHANNEL-LAST [Cout, Kh*Kw*Cin] -- THE dangerous line
      in this feature (see emit.emit_conv_chain_module's own docstring
      for why, and its own named/mutation test for the consequence of
      getting it backwards).
    - `x_name`/`x_shape`/`x_zero_point` are exactly load_qlinear_conv's
      own returns, for layer 0's activation.
    - `conv_params[i]` is layer i's own (kh, kw, stride_h, stride_w,
      pad_top, pad_bottom, pad_left, pad_right, dilation_h, dilation_w),
      for every i from 0 to len(conv_nodes) - 1 -- `conv_params[0]` is
      passed to `im2col_nchw` for layer 0's own Python-side patches
      exactly as load_qlinear_conv's own return is, and the WHOLE list is
      passed straight through to emit.emit_conv_chain_module (whose own
      `conv_params[0]` entry it does not read, by that function's own
      documented convention).
    - `trailing_requantize`/`per_channel_requantize`/`bias`/
      `uint8_output_shifted` are exactly load_qlinear_conv's own returns,
      for the LAST layer only -- pass them straight through to
      emit.emit_conv_chain_module's own `last_trailing_requantize`/
      `last_per_channel_requantize`/`last_bias` parameters.
    - `bridge_scales` has `len(conv_nodes) - 1` entries, bridge i's real
      positive cim.requantize scale -- see this section's own module
      header for the derivation and why no cross-layer validation is
      needed.

    Raises Refusal for anything outside this function's own scope (see
    this section's own module header): fewer than two QLinearConv nodes,
    any non-conv node in the graph, a branching/cyclic/disconnected
    "chain", a non-zero y_zero_point/per-channel w_scale/bias on any
    layer but the last, or a non-zero x_zero_point on any layer but the
    first.
    """
    from onnx import numpy_helper

    _check_opset(model)
    graph = model.graph
    initializers = _tensor_names(model)

    convs = [n for n in graph.node if n.op_type == ACCEPTED_CONV_OP]
    if len(convs) < 2:
        raise Refusal(
            f"found {len(convs)} {ACCEPTED_CONV_OP} node(s); load_conv_chain "
            f"handles a chain of two or more. Use load_qlinear_conv for a "
            f"standalone convolution, or load_conv_matmul_chain for a "
            f"convolution feeding MatMulInteger layers, instead. Nothing "
            f"emitted.")

    others = [n for n in graph.node if n.op_type != ACCEPTED_CONV_OP]
    if others:
        kinds = sorted({n.op_type for n in others})
        raise Refusal(
            f"graph also contains {', '.join(kinds)}. A chain of "
            f"{ACCEPTED_CONV_OP} nodes needs no bridge node between layers "
            f"(unlike a {ACCEPTED_CONV_OP} chained into {ACCEPTED_OP}, or a "
            f"chain of {ACCEPTED_OP} layers) -- conv_i's own \"Y\" feeds "
            f"conv_(i+1)'s own \"X\" directly, both natively [N, C, H, W] "
            f"per the ONNX spec. Nothing emitted.")

    # Chain-ordering discovery via producer/consumer edges, not node-list
    # order.
    producer = {n.output[0]: n for n in convs}
    consumers = {}
    for n in graph.node:
        for inp in n.input:
            if inp:
                consumers.setdefault(inp, []).append(n)

    starts = [n for n in convs if n.input[0] not in producer]
    if len(starts) != 1:
        raise Refusal(
            f"found {len(starts)} {ACCEPTED_CONV_OP} node(s) whose own "
            f"activation is not another {ACCEPTED_CONV_OP}'s output; a "
            f"single linear chain has exactly one such node (its own "
            f"first layer) -- {len(starts)} means either a disconnected "
            f"graph or more than one chain. Nothing emitted.")

    chain = [starts[0]]
    seen_ids = {id(starts[0])}
    current = starts[0]
    while len(chain) < len(convs):
        out = current.output[0]
        readers = consumers.get(out, [])
        conv_readers = [r for r in readers if r.op_type == ACCEPTED_CONV_OP]
        if (len(readers) != 1 or len(conv_readers) != 1
                or conv_readers[0].input[0] != out):
            raise Refusal(
                f"node '{current.name or ACCEPTED_CONV_OP}''s own output "
                f"is not read as exactly one other {ACCEPTED_CONV_OP}'s "
                f"own activation ({len(readers)} total reader(s) found). "
                f"A chain must be linear -- no branching, no extra "
                f"consumers, no non-conv reader. Nothing emitted.")
        nxt = conv_readers[0]
        if id(nxt) in seen_ids:
            raise Refusal(
                f"the {ACCEPTED_CONV_OP} chain forms a cycle. Nothing "
                f"emitted.")
        chain.append(nxt)
        seen_ids.add(id(nxt))
        current = nxt

    if len(graph.output) != 1:
        raise Refusal(
            f"graph has {len(graph.output)} outputs; a chain must end in "
            f"exactly one. Nothing emitted.")
    if graph.output[0].name != chain[-1].output[0]:
        raise Refusal(
            f"the graph's output is not the chain's own last "
            f"{ACCEPTED_CONV_OP}'s own output. Nothing emitted.")

    n_layers = len(chain)
    weights2d = []
    conv_params = []
    bridge_scales = []
    x_name = None
    x_shape = None
    x_zero_point = 0
    trailing_requantize = None
    per_channel_requantize = None
    bias = None
    uint8_output_shifted = False
    prev_cout = None

    for i, node in enumerate(chain):
        is_first = i == 0
        is_last = i == n_layers - 1
        where = f"node '{node.name or ACCEPTED_CONV_OP}' (layer {i})"

        if len(node.input) < 8:
            raise Refusal(
                f"{ACCEPTED_CONV_OP} has {len(node.input)} inputs; x, "
                f"x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale "
                f"and y_zero_point are all mandatory per the ONNX spec. "
                f"Nothing emitted.", where=where)
        (x_i, x_scale_name, x_zp_name, w_name, w_scale_name, w_zp_name,
         y_scale_name, y_zp_name) = node.input[:8]
        bias_name = node.input[8] if len(node.input) > 8 else ""

        if not is_last and bias_name:
            raise Refusal(
                f"this node has a bias operand '{bias_name}', but only "
                f"the LAST layer of a {ACCEPTED_CONV_OP} chain may have "
                f"one -- an intermediate layer's own bridge is "
                f"emit_conv_chain_module's unconditional scalar "
                f"cim.requantize, with no cim.reduce_partial step. "
                f"Compile this chain up to (and not including) this "
                f"layer instead, or drop the bias. Nothing emitted.",
                where=where)
        if w_name not in initializers:
            raise Refusal(
                f"the weight operand '{w_name}' is not a constant "
                f"initializer. Nothing emitted.", where=where)
        if x_i in initializers:
            raise Refusal(
                f"both operands are constant initializers. Nothing "
                f"emitted.", where=where)

        w_init = next(t for t in graph.initializer if t.name == w_name)
        weight = numpy_helper.to_array(w_init)
        if weight.dtype != np.int8:
            raise Refusal(
                f"the weight operand '{w_name}' has dtype {weight.dtype}. "
                f"Nothing emitted.", where=where)
        if weight.ndim != 4:
            raise Refusal(
                f"the weight operand '{w_name}' has rank {weight.ndim}; a "
                f"2-D convolution's kernel is rank 4 [Cout, Cin, Kh, Kw]. "
                f"Nothing emitted.", where=where)
        cout, cin, kh, kw = weight.shape

        if not is_first and cin != prev_cout:
            raise Refusal(
                f"layer {i}'s weight has Cin={cin}, but layer {i - 1}'s "
                f"weight has Cout={prev_cout} -- one conv's own output "
                f"channel count must match the next conv's own input "
                f"channel count. Nothing emitted.", where=where)

        x_scale = _positive_scalar_scale(graph, x_scale_name, node.name,
                                         "x_scale")
        w_scale = _positive_weight_scale(graph, w_scale_name, node.name,
                                         "w_scale", cout)
        y_scale = _positive_scalar_scale(graph, y_scale_name, node.name,
                                         "y_scale")
        layer_x_zp, _x_zp_dtype = _scalar_zero_point(
            graph, x_zp_name, node.name, "x_zero_point",
            allow_nonzero=is_first)
        _weight_zero_point_must_be_zero(graph, w_zp_name, node.name, cout)
        layer_y_zp, y_zp_dtype = _scalar_zero_point(
            graph, y_zp_name, node.name, "y_zero_point",
            allow_nonzero=is_last)

        if not is_last and isinstance(w_scale, np.ndarray):
            raise Refusal(
                f"w_scale has {w_scale.size} elements; only the LAST "
                f"layer of a {ACCEPTED_CONV_OP} chain may have a "
                f"per-channel w_scale -- an intermediate layer's own "
                f"bridge is one unconditional scalar cim.requantize, not "
                f"the N-way per-channel split a standalone conv's own "
                f"per_channel_requantize uses. Nothing emitted.",
                where=where)
        if not is_last and y_zp_dtype != np.int8:
            raise Refusal(
                f"y_zero_point is {y_zp_dtype}; an intermediate layer of a "
                f"{ACCEPTED_CONV_OP} chain must declare an int8 output. "
                f"The LAST layer can accept uint8 (load_qlinear_conv's own "
                f"uint8_output_shifted note) because it documents a "
                f"caller-side +128 shift on the FINAL printed result -- "
                f"but there is no caller here to apply that shift before "
                f"the next layer reads this value. Nothing emitted.",
                where=where)

        (dilation_h, dilation_w, pad_top, pad_bottom, pad_left, pad_right,
         stride_h, stride_w) = _conv_geometry(node, weight, where)
        conv_params.append((kh, kw, stride_h, stride_w, pad_top, pad_bottom,
                           pad_left, pad_right, dilation_h, dilation_w))

        if is_first:
            x_info = next((v for v in graph.input if v.name == x_i), None)
            if x_info is None:
                raise Refusal(
                    f"the activation operand '{x_i}' is neither a graph "
                    f"input nor a constant. Nothing emitted.", where=where)
            if x_info.type.tensor_type.elem_type not in (_INT8, _UINT8):
                raise Refusal(
                    f"the activation operand '{x_i}' is not int8 or "
                    f"uint8. Nothing emitted.", where=where)
            shp = _shape_of(x_info)
            if shp is None:
                raise Refusal(
                    f"the activation operand '{x_i}' has a symbolic or "
                    f"unknown shape. Nothing emitted.", where=where)
            if len(shp) != 4:
                raise Refusal(
                    f"the activation operand '{x_i}' has rank {len(shp)}; "
                    f"a 2-D convolution's input is rank 4 [N, Cin, H, W]. "
                    f"Nothing emitted.", where=where)
            if shp[1] != cin:
                raise Refusal(
                    f"activation has Cin={shp[1]} but the weight is "
                    f"{list(weight.shape)} (Cin={cin}). Nothing emitted.",
                    where=where)
            x_name = x_i
            x_shape = tuple(shp)
            x_zero_point = layer_x_zp
            weights2d.append(
                np.ascontiguousarray(weight.reshape(cout, cin * kh * kw)))
        else:
            # Channel-last: layer i's own activation is layer i-1's own
            # bridged [M, Cout] output, NHWC by construction of how
            # emit_conv_chain_module's own gather lays out patches columns
            # (tap-major, channel-minor within each tap) -- see this
            # section's own module header, and emit_conv_chain_module's
            # own docstring, for why getting this backwards is silently
            # wrong rather than a shape error.
            weights2d.append(np.ascontiguousarray(
                weight.transpose(0, 2, 3, 1).reshape(cout, kh * kw * cin)))

        if is_last:
            uint8_output_shifted = y_zp_dtype == np.uint8
            zp_for_module = (layer_y_zp - 128 if uint8_output_shifted
                            else layer_y_zp)
            if isinstance(w_scale, np.ndarray):
                derived_scales = y_scale / (x_scale * w_scale)
                trailing_requantize = None
                per_channel_requantize = [(float(s), zp_for_module)
                                          for s in derived_scales]
            else:
                derived_scale = y_scale / (x_scale * w_scale)
                trailing_requantize = (derived_scale, zp_for_module, 8)
                per_channel_requantize = None
            if bias_name:
                bias = np.asarray(_initializer_or_refuse(
                    graph, bias_name, node.name, "B")).astype(np.int64)
                if bias.shape != (cout,):
                    raise Refusal(
                        f"bias 'B' has shape {list(bias.shape)}; expected "
                        f"a single length-Cout ({cout}) int32 array, one "
                        f"value per output channel. Nothing emitted.",
                        where=where)
                bias = bias.astype(np.int32)
        else:
            bridge_scales.append(y_scale / (x_scale * w_scale))

        prev_cout = cout

    return (chain, weights2d, x_name, x_shape, conv_params, x_zero_point,
           trailing_requantize, per_channel_requantize, bias,
           uint8_output_shifted, bridge_scales)


def _validate_conv_activation(activation, x_shape, x_zero_point=0):
    """The supplied activation must be exactly `x_shape` ([N, Cin, H, W]).

    Returns it as int8, with `x_zero_point` already subtracted -- exact
    whenever w_zero_point == 0 (always true here; see load_qlinear_conv's
    own 'A REAL MODEL FORCED A SECOND LOOK' note for why). Refuses,
    naming the actual out-of-range values, if the shifted result does not
    fit signed int8, rather than silently wrapping -- the default
    `x_zero_point=0` makes this exactly the old (pre-asymmetric-
    quantization) range check, so nothing changes for a symmetric model.
    """
    activation = np.asarray(activation)
    if tuple(activation.shape) != tuple(x_shape):
        raise Refusal(
            f"the supplied activation has shape {list(activation.shape)}; "
            f"the model's convolution input is {list(x_shape)}. Nothing "
            f"emitted.")
    shifted = activation.astype(np.int64) - int(x_zero_point)
    lo, hi = int(shifted.min()), int(shifted.max())
    if lo < -128 or hi > 127:
        raise Refusal(
            f"the supplied activation, after subtracting x_zero_point "
            f"({x_zero_point}), has values in [{lo}, {hi}], which does not "
            f"fit signed int8. Nothing emitted.")
    return shifted.astype(np.int8)


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

    Dispatches on how many MatMulInteger and QLinearConv nodes the graph
    has. One or more QLinearConv AND one or more MatMulInteger nodes goes
    through load_conv_matmul_chain, which itself refuses anything but
    exactly one QLinearConv (the conv is always layer 0 -- a SECOND
    convolution feeding further MatMulInteger layers is out of scope for
    both loaders; see load_conv_matmul_chain's own module section header).
    Zero MatMulInteger nodes and two or more QLinearConv nodes goes
    through load_conv_chain (a real, direct conv-to-conv chain -- see its
    own module section header for why it needs no bridge node the way
    every other chain this front end accepts does). Zero MatMulInteger
    nodes and a single QLinearConv goes through load_qlinear_conv
    (standalone). Two or more MatMulInteger nodes and no QLinearConv go
    through load_matmul_chain; exactly one of either goes through the
    single-layer paths (load_matmul_integer, load_qlinear_conv). Anything
    else -- more than one QLinearConv, or stray Cast/QuantizeLinear nodes
    around a single MatMulInteger -- is refused by the relevant loader's
    own "graph also contains ..." check, which already says the right
    thing for that case.
    """
    from onnx import TensorProto  # noqa: F401 (documents op_type below)

    matmul_count = sum(1 for n in model.graph.node
                       if n.op_type == ACCEPTED_OP)
    conv_count = sum(1 for n in model.graph.node
                     if n.op_type == ACCEPTED_CONV_OP)

    if conv_count >= 1 and matmul_count >= 1:
        (conv_node, matmul_nodes, weight2d, x_name, x_shape, conv_params,
         x_zero_point, matmul_weights, bridge_scales) = (
            load_conv_matmul_chain(model))
        activation = _validate_conv_activation(activation, x_shape,
                                               x_zero_point)

        patches, (n_batch, out_h, out_w) = im2col_nchw(activation,
                                                        *conv_params)

        taken = set()
        weight_syms = [sanitize_symbol(conv_node.input[3], taken)]
        weight_syms += [sanitize_symbol(n.input[1], taken)
                        for n in matmul_nodes]
        act_sym = sanitize_symbol(x_name, taken)

        header = provenance(model_path, model_bytes, activation_source)
        header += (
            f"//   conv im2col (layer 0): N={n_batch} OutH={out_h} "
            f"OutW={out_w} -- the whole chain's printed output rows stay "
            f"in (n, oh, ow) row-major order (emit_chain_module threads M "
            f"through every layer unchanged), reshape to "
            f"[N, OutH, OutW, Cout] then transpose(0, 3, 1, 2) for ONNX's "
            f"own [N, Cout, OutH, OutW] layout.\n")
        return emit_chain_module(
            [weight2d] + matmul_weights, patches, weight_syms=weight_syms,
            act_sym=act_sym, header=header, scales=bridge_scales)

    if matmul_count == 0 and conv_count >= 2:
        (conv_nodes, weights2d, x_name, x_shape, conv_params, x_zero_point,
         trailing_requantize, per_channel_requantize, bias,
         uint8_output_shifted, bridge_scales) = load_conv_chain(model)
        activation = _validate_conv_activation(activation, x_shape,
                                               x_zero_point)

        patches0, out_shape0 = im2col_nchw(activation, *conv_params[0])
        n_batch, out_h, out_w = out_shape0

        taken = set()
        weight_syms = [sanitize_symbol(n.input[3], taken) for n in conv_nodes]
        act_sym = sanitize_symbol(x_name, taken)

        header = provenance(model_path, model_bytes, activation_source)
        header += (
            f"//   conv im2col (layer 0): N={n_batch} OutH={out_h} "
            f"OutW={out_w} -- the whole chain's printed output rows stay "
            f"in (n, oh, ow) row-major order (emit_conv_chain_module "
            f"threads M through every layer, exactly like "
            f"emit_chain_module), reshape to [N, OutH, OutW, Cout] then "
            f"transpose(0, 3, 1, 2) for ONNX's own [N, Cout, OutH, OutW] "
            f"layout.\n")
        if uint8_output_shifted:
            header += (
                f"//   NOTE: the last layer's y_zero_point is uint8, so "
                f"its true output is uint8 too -- but cim.requantize's "
                f"clamp is SIGNED. Every printed value below is "
                f"(true_uint8_value - 128), EXACTLY (see "
                f"load_qlinear_conv's own uint8_output_shifted note): add "
                f"128 back to recover the real QLinearConv output.\n")
        return emit_conv_chain_module(
            weights2d, conv_params, patches0, out_shape0,
            weight_syms=weight_syms, act_sym=act_sym, header=header,
            scales=bridge_scales, last_bias=bias,
            last_trailing_requantize=trailing_requantize,
            last_per_channel_requantize=per_channel_requantize)

    if matmul_count == 0 and conv_count >= 1:
        (node, weight2d, x_name, x_shape, conv_params, x_zero_point,
         trailing_requantize, per_channel_requantize, bias,
         uint8_output_shifted) = load_qlinear_conv(model)
        activation = _validate_conv_activation(activation, x_shape,
                                               x_zero_point)

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
        if uint8_output_shifted:
            header += (
                f"//   NOTE: this node's y_zero_point is uint8, so its true "
                f"output is uint8 too -- but cim.requantize's clamp is "
                f"SIGNED. Every printed value below is (true_uint8_value - "
                f"128), EXACTLY (see load_qlinear_conv's own "
                f"uint8_output_shifted note): add 128 back to recover the "
                f"real QLinearConv output.\n")
        return emit_module(
            weight2d, patches, weight_sym=weight_sym, act_sym=act_sym,
            header=header, trailing_requantize=trailing_requantize,
            per_channel_requantize=per_channel_requantize, bias=bias)

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
