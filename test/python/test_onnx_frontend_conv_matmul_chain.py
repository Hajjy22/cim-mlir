"""QLinearConv feeding one or more MatMulInteger layers -- the first step
of "chain convolution layers" (docs/roadmap.md's plan): the smallest real
capability step beyond what already exists, since layer 0's activation is
still the literal graph input, so the existing, unchanged Python
`im2col_nchw` call already produces exactly what `emit_chain_module`
already consumes for layer 0. See
`cim_frontend.onnx_import.load_conv_matmul_chain`'s own module section
header for the full design.

THE Transpose -> Reshape BRIDGE IS LOAD-BEARING, NOT COSMETIC
------------------------------------------------------------------
Confirmed by hand before this loader was written: ONNX's own
`MatMulInteger` reference kernel does N-D numpy-broadcast matmul rather
than requiring 2-D operands, so a graph that wires a QLinearConv's raw
`[N, Cout, OutH, OutW]` output directly into a `MatMulInteger` node does
NOT error -- it silently contracts whatever axes happen to align (e.g.
`OutW` against `Cout`, if they happen to be equal), a genuinely different,
wrong function. `Transpose(perm=[0, 2, 3, 1]) -> Reshape([M, Cout])`
between the conv and the first matmul is therefore REQUIRED in the graph,
not merely tolerated, to make the .onnx file itself a faithful,
standards-compliant statement of the same function this compiler emits --
this is why `onnx_fixtures.conv_matmul_chain_model` always inserts it, and
why `load_conv_matmul_chain` refuses anything else in that position.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python"))

pytest.importorskip("onnx", reason="the ONNX front end's oracle")

import onnx  # noqa: E402
from onnx import helper as onnx_helper  # noqa: E402
from onnx import numpy_helper  # noqa: E402

from cim_frontend.onnx_import import import_model  # noqa: E402
from cim_frontend.refusal import Refusal  # noqa: E402
from onnx_fixtures import conv_matmul_chain_model, onnx_reference_eval  # noqa: E402
from test_numerical_differential import compile_and_run, PipelineError  # noqa: E402

SEED = int(os.environ.get("CIM_TEST_SEED", "0x5eed1234abcd"), 0)


def _run(model, act, cim_opt, cim_run):
    want = onnx_reference_eval(model, act, act_name="X")
    text = import_model(model, act)
    try:
        outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                     source=text)
    except PipelineError as err:
        pytest.fail(f"the imported conv->matmul chain failed to compile "
                    f"or run: {err}")
    return np.asarray(outputs), want


# --- correctness -------------------------------------------------------

def test_a_conv_feeding_one_matmul_matches_the_reference(cim_opt, cim_run):
    cout, cin, kh, kw = 3, 2, 2, 2
    rng = np.random.default_rng(SEED + 30)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 4, 4)
    mm_w = rng.integers(-4, 5, size=(cout, 5), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_conv_feeding_two_matmuls_with_stride_pad_and_a_real_scale(
        cim_opt, cim_run):
    # Exercises the three parameters a conv->matmul chain most needs to
    # get right together: a strided, padded conv (so the Reshape bridge's
    # M = N*OutH*OutW must be derived correctly, not just copied from a
    # stride=1/pad=0 case) and a real (non-1.0) bridge scale between the
    # two MatMulInteger layers, exactly like test_onnx_frontend_chain.py's
    # own real-scale coverage.
    cout, cin, kh, kw = 3, 2, 2, 2
    rng = np.random.default_rng(SEED + 31)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 5, 5)
    mm_w0 = rng.integers(-4, 5, size=(cout, 4), dtype=np.int64).astype(np.int8)
    mm_w1 = rng.integers(-4, 5, size=(4, 6), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(
        conv_w, x_shape, [mm_w0, mm_w1], strides=(2, 2), pads=(1, 1, 1, 1),
        bridge_scales=[3.0])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_batched_conv_matmul_chain_matches_the_reference(cim_opt, cim_run):
    # N > 1: M = N*OutH*OutW threads through the whole chain unchanged
    # (emit_chain_module's own note), so a real batch exercises that the
    # Reshape bridge's M is derived from N too, not just OutH*OutW.
    cout, cin, kh, kw = 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 32)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (2, cin, 3, 3)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_wrong_weight_in_the_matmul_layer_would_actually_be_caught(
        cim_opt, cim_run):
    # Anti-vacuity: a perturbation in the MATMUL layer (not the conv) must
    # still change the compiled result, proving the bridge actually wires
    # the conv's output into the matmul rather than the test only
    # exercising the conv in isolation.
    cout, cin, kh, kw = 3, 2, 2, 2
    rng = np.random.default_rng(SEED + 33)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 4, 4)
    mm_w = rng.integers(-4, 5, size=(cout, 5), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    base_model = conv_matmul_chain_model(conv_w, x_shape, [mm_w])
    base = onnx_reference_eval(base_model, act, act_name="X")

    bumped_mm_w = mm_w.copy()
    bumped_mm_w[0, 0] = np.int8(int(bumped_mm_w[0, 0]) ^ 0x5A)
    bumped_model = conv_matmul_chain_model(conv_w, x_shape, [bumped_mm_w])
    want = onnx_reference_eval(bumped_model, act, act_name="X")
    assert not np.array_equal(base, want), "the perturbation changed nothing"

    outputs, _ = compile_and_run(
        cim_opt, cim_run, None, None,
        source=import_model(bumped_model, act))
    assert np.array_equal(np.asarray(outputs), want)


# --- refusals ------------------------------------------------------------

def test_refuses_a_missing_transpose_reshape_bridge():
    # The direct-edge shape this loader deliberately does NOT accept --
    # see this file's own module docstring for why wiring the conv's raw
    # output straight into MatMulInteger would be a real, silent wrong
    # answer under ONNX's own MatMulInteger semantics, not merely
    # unsupported.
    cout, cin, kh, kw = 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 34)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 3, 3)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], check=False)
    # Splice the graph: point the matmul directly at the conv's raw
    # output, dropping the Transpose/Reshape nodes entirely.
    nodes = [n for n in model.graph.node if n.op_type != "Transpose"
            and n.op_type != "Reshape"]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.node[1].input[0] = "conv_y"  # mm0's A input -> conv's Y

    # Matched against the specific "wrong reader" wording, not a bare
    # substring like "Transpose": mutation-testing this test (disabling
    # the reader-count/op-type check it targets) found that a LOOSER match
    # would pass for the wrong reason -- the next check down (the perm
    # check) also happens to print the literal word "Transpose" while
    # actually describing a mislabeled MatMulInteger node, so a substring
    # match alone does not prove the check under test actually fired.
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="not read by exactly one Transpose"):
        import_model(model, act)


def test_refuses_a_wrong_transpose_permutation():
    cout, cin, kh, kw = 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 35)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 3, 3)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], check=False)
    transpose = next(n for n in model.graph.node if n.op_type == "Transpose")
    for a in transpose.attribute:
        if a.name == "perm":
            a.ints[:] = [0, 1, 2, 3]  # identity, not NCHW -> NHWC

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match=r"perm=\[0, 1, 2, 3\]"):
        import_model(model, act)


def test_refuses_a_wrong_reshape_target_shape():
    cout, cin, kh, kw = 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 36)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 3, 3)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], check=False)
    for init in model.graph.initializer:
        if init.name == "reshape0_shape":
            init.CopyFrom(numpy_helper.from_array(
                np.array([1, 1], dtype=np.int64), name="reshape0_shape"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="targets"):
        import_model(model, act)


def test_refuses_more_than_one_qlinear_conv_in_a_chain():
    cout, cin, kh, kw = 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 37)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 3, 3)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], check=False)
    extra_conv = onnx_helper.make_node(
        "QLinearConv",
        ["X", "conv_xs", "conv_xzp", "conv_W", "conv_ws", "conv_wzp",
         "conv_ys", "conv_yzp"],
        ["conv2_y"], name="conv1", strides=[1, 1], pads=[0, 0, 0, 0])
    model.graph.node.insert(0, extra_conv)

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="found 2 QLinearConv"):
        import_model(model, act)


def test_refuses_a_nonzero_conv_y_zero_point():
    cout, cin, kh, kw = 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 38)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 3, 3)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], check=False)
    for init in model.graph.initializer:
        if init.name == "conv_yzp":
            init.CopyFrom(numpy_helper.from_array(
                np.array(5, dtype=np.int8), name="conv_yzp"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="symmetric"):
        import_model(model, act)


def test_refuses_a_per_channel_conv_w_scale_in_a_chain():
    cout, cin, kh, kw = 3, 2, 2, 2
    rng = np.random.default_rng(SEED + 39)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 3, 3)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], check=False)
    for init in model.graph.initializer:
        if init.name == "conv_ws":
            init.CopyFrom(numpy_helper.from_array(
                np.full(cout, 1.0, dtype=np.float32), name="conv_ws"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="single scalar w_scale"):
        import_model(model, act)


def test_refuses_a_conv_bias_in_a_chain():
    cout, cin, kh, kw = 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 40)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 3, 3)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], check=False)
    for node in model.graph.node:
        if node.name == "conv0":
            model.graph.initializer.append(numpy_helper.from_array(
                np.zeros(cout, dtype=np.int32), name="conv_bias"))
            node.input.append("conv_bias")

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="bias operand"):
        import_model(model, act)


def test_refuses_a_uint8_conv_output_in_a_chain():
    from onnx import TensorProto

    cout, cin, kh, kw = 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 41)
    conv_w = rng.integers(-4, 5, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 3, 3)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], check=False)
    for init in model.graph.initializer:
        if init.name == "conv_yzp":
            init.CopyFrom(numpy_helper.from_array(
                np.array(0, dtype=np.uint8), name="conv_yzp"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="int8 output"):
        import_model(model, act)
