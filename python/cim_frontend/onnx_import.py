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

from .emit import emit_module, sanitize_symbol
from .refusal import Refusal

# MatMulInteger was introduced at opset 10.
MIN_OPSET = 10

ACCEPTED_OP = "MatMulInteger"


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
            f"found {len(matmuls)} {ACCEPTED_OP} nodes; this version imports a "
            f"single one. Chained layers are not yet supported. Nothing "
            f"emitted.")

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
    if m != 1:
        raise Refusal(
            f"the activation has {m} rows. The v0.1 contract is matrix-VECTOR "
            f"(a single output row); cim-partition refuses more and would "
            f"leave the matmul unoffloaded. Nothing emitted.", where=where)
    if weight.shape[0] != k:
        raise Refusal(
            f"shape mismatch: activation is [{m}, {k}] but the weight is "
            f"{list(weight.shape)}; MatMulInteger needs B to be [K, N] with "
            f"K = {k}. Nothing emitted.", where=where)

    # THE transpose. ONNX B is [K, N]; cim.mvm wants output-major [N, K].
    return node, np.ascontiguousarray(weight.T), a_name


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


def import_model(model, activation, model_path="<model>", model_bytes=b"",
                 activation_source="<supplied>"):
    """An ONNX model plus an activation vector -> MLIR text."""
    node, weight, act_name = load_matmul_integer(model)

    activation = np.asarray(activation)
    if activation.ndim == 2 and activation.shape[0] == 1:
        activation = activation[0]
    if activation.ndim != 1:
        raise Refusal(
            f"the supplied activation has shape {list(activation.shape)}; "
            f"expected [1, K] or [K]. Nothing emitted.")
    if activation.shape[0] != weight.shape[1]:
        raise Refusal(
            f"the supplied activation has length {activation.shape[0]} but the "
            f"model's K is {weight.shape[1]}. Nothing emitted.")
    if activation.dtype != np.int8:
        lo, hi = int(activation.min()), int(activation.max())
        if lo < -128 or hi > 127:
            raise Refusal(
                f"the supplied activation has values in [{lo}, {hi}], which "
                f"does not fit int8. Wrapping them silently would compute with "
                f"different numbers than were supplied. Nothing emitted.")
        activation = activation.astype(np.int8)

    taken = set()
    weight_sym = sanitize_symbol(node.input[1], taken)
    act_sym = sanitize_symbol(act_name, taken)

    header = provenance(model_path, model_bytes, activation_source)
    return emit_module(weight, activation, weight_sym=weight_sym,
                       act_sym=act_sym, header=header)
