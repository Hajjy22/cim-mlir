"""A chain of two or more QLinearConv nodes feeding one or more
MatMulInteger layers -- PR D of the "chain convolution layers" plan: a
real conv-stem-feeding-an-FC-head chain, the realistic full CNN shape.
Built by composing `load_conv_chain`'s own conv-chain machinery (PR C)
with `load_conv_matmul_chain`'s own conv-to-matmul Transpose/Reshape
bridge (PR B) -- see `cim_frontend.onnx_import.load_conv_chain_matmul_
chain`'s own module section header for the full design.

WHY EVERY CONV LAYER HERE STAYS RESTRICTED, INCLUDING THE LAST
------------------------------------------------------------------
Unlike `load_conv_chain`'s own last layer (which gets full generality --
asymmetric output, per-channel w_scale, a bias -- because it IS the
graph's own declared output there), NO conv layer here is ever the
graph's own output: even the chain's last conv feeds the Transpose/
Reshape bridge into a matmul, never the graph directly. So every layer
keeps `load_conv_matmul_chain`'s own restriction on its single conv,
generalized uniformly: `y_zero_point == 0`, a scalar `w_scale`, no bias.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python"))

pytest.importorskip("onnx", reason="the ONNX front end's oracle")

from onnx import numpy_helper  # noqa: E402

from cim_frontend.onnx_import import import_model  # noqa: E402
from cim_frontend.refusal import Refusal  # noqa: E402
from onnx_fixtures import (conv_chain_matmul_chain_model,  # noqa: E402
                          onnx_reference_eval)
from test_numerical_differential import compile_and_run, PipelineError  # noqa: E402

SEED = int(os.environ.get("CIM_TEST_SEED", "0x5eed1234abcd"), 0)


def _run(model, act, cim_opt, cim_run):
    want = onnx_reference_eval(model, act, act_name="X")
    text = import_model(model, act)
    try:
        outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                     source=text)
    except PipelineError as err:
        pytest.fail(f"the imported conv-chain-to-matmul chain failed to "
                    f"compile or run: {err}")
    return np.asarray(outputs), want


# --- correctness -------------------------------------------------------

def test_a_two_layer_conv_chain_feeding_one_matmul_matches_the_reference(
        cim_opt, cim_run):
    rng = np.random.default_rng(SEED + 70)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model([w0, w1], x_shape, [mm_w])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_conv_chain_with_stride_pad_and_dilation_feeding_two_matmuls(
        cim_opt, cim_run):
    # Exercises a strided/padded/dilated interior conv layer (the tap-
    # gather geometry) TOGETHER with a real (non-1.0) inter-matmul bridge
    # scale (load_matmul_chain's own real-scale coverage), in one graph.
    rng = np.random.default_rng(SEED + 71)
    w0 = rng.integers(-4, 5, size=(3, 2, 3, 3), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 7, 7)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w0 = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)
    mm_w1 = rng.integers(-4, 5, size=(4, 6), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w0, mm_w1],
        strides=[(1, 1), (2, 2)], pads=[(0, 0, 0, 0), (1, 1, 1, 1)],
        dilations=[(1, 1), (2, 2)], bridge_scales=[3.0])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_batched_conv_chain_matmul_chain_matches_the_reference(
        cim_opt, cim_run):
    rng = np.random.default_rng(SEED + 72)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (2, 2, 4, 4)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model([w0, w1], x_shape, [mm_w])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_three_layer_conv_chain_feeding_a_matmul_matches_the_reference(
        cim_opt, cim_run):
    rng = np.random.default_rng(SEED + 73)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(4, 3, 2, 2), dtype=np.int64).astype(np.int8)
    w2 = rng.integers(-4, 5, size=(2, 4, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 5), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model([w0, w1, w2], x_shape, [mm_w])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_wrong_weight_in_an_interior_conv_layer_would_actually_be_caught(
        cim_opt, cim_run):
    # Anti-vacuity: a perturbation in an INTERIOR conv layer of a 3-conv
    # + 1-matmul chain must still change the compiled result.
    rng = np.random.default_rng(SEED + 74)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(4, 3, 2, 2), dtype=np.int64).astype(np.int8)
    w2 = rng.integers(-4, 5, size=(2, 4, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 5), dtype=np.int64).astype(np.int8)

    base_model = conv_chain_matmul_chain_model([w0, w1, w2], x_shape, [mm_w])
    base = onnx_reference_eval(base_model, act, act_name="X")

    bumped_w1 = w1.copy()
    bumped_w1[0, 0, 0, 0] = np.int8(int(bumped_w1[0, 0, 0, 0]) ^ 0x5A)
    bumped_model = conv_chain_matmul_chain_model(
        [w0, bumped_w1, w2], x_shape, [mm_w])
    unreshaped_want = onnx_reference_eval(bumped_model, act, act_name="X")
    assert not np.array_equal(base, unreshaped_want), (
        "the perturbation changed nothing")

    outputs, want = _run(bumped_model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want)


def test_a_wrong_weight_in_the_matmul_layer_would_actually_be_caught(
        cim_opt, cim_run):
    # Anti-vacuity, the other direction: a perturbation in the MATMUL
    # layer (not any conv) must still change the compiled result.
    rng = np.random.default_rng(SEED + 75)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)

    base_model = conv_chain_matmul_chain_model([w0, w1], x_shape, [mm_w])
    base = onnx_reference_eval(base_model, act, act_name="X")

    bumped_mm_w = mm_w.copy()
    bumped_mm_w[0, 0] = np.int8(int(bumped_mm_w[0, 0]) ^ 0x5A)
    bumped_model = conv_chain_matmul_chain_model(
        [w0, w1], x_shape, [bumped_mm_w])
    unreshaped_want = onnx_reference_eval(bumped_model, act, act_name="X")
    assert not np.array_equal(base, unreshaped_want), (
        "the perturbation changed nothing")

    outputs, want = _run(bumped_model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want)


def test_a_relu_between_two_convs_feeding_a_matmul_matches_the_reference(
        cim_opt, cim_run):
    # Relu on the interior conv-conv bridge, reused unchanged from
    # load_conv_chain's own acceptance (see _discover_conv_chain) --
    # this loader's own matmul tail is untouched by it.
    rng = np.random.default_rng(SEED + 85)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], relu_after={0})
    no_relu_model = conv_chain_matmul_chain_model([w0, w1], x_shape, [mm_w])
    with_relu = onnx_reference_eval(model, act, act_name="X")
    without_relu = onnx_reference_eval(no_relu_model, act, act_name="X")
    assert not np.array_equal(with_relu, without_relu), (
        "fixture assumption broken: Relu made no difference")

    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


# --- refusals ------------------------------------------------------------

def test_refuses_a_missing_transpose_reshape_bridge_after_the_chain():
    rng = np.random.default_rng(SEED + 76)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], check=False)
    nodes = [n for n in model.graph.node if n.op_type not in
            ("Transpose", "Reshape")]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    # mm0's own A input -> the last conv's raw output directly.
    for node in model.graph.node:
        if node.name == "mm0":
            node.input[0] = "L1_Y"

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="not read by exactly one Transpose"):
        import_model(model, act)


def test_refuses_a_nonzero_y_zero_point_on_the_last_conv_layer():
    rng = np.random.default_rng(SEED + 77)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], check=False)
    for init in model.graph.initializer:
        if init.name == "L1_yzp":
            init.CopyFrom(numpy_helper.from_array(
                np.array(5, dtype=np.int8), name="L1_yzp"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="symmetric"):
        import_model(model, act)


def test_refuses_a_per_channel_w_scale_on_the_last_conv_layer():
    rng = np.random.default_rng(SEED + 78)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)
    mm_w = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], check=False)
    for init in model.graph.initializer:
        if init.name == "L1_ws":
            init.CopyFrom(numpy_helper.from_array(
                np.full(2, 1.0, dtype=np.float32), name="L1_ws"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="requires a single scalar w_scale"):
        import_model(model, act)


def test_refuses_a_bias_on_the_last_conv_layer():
    rng = np.random.default_rng(SEED + 79)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], check=False)
    for node in model.graph.node:
        if node.name == "conv1":
            model.graph.initializer.append(numpy_helper.from_array(
                np.zeros(2, dtype=np.int32), name="L1_B"))
            node.input.append("L1_B")

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="bias operand"):
        import_model(model, act)


def test_refuses_a_nonzero_x_zero_point_on_a_non_first_conv_layer():
    rng = np.random.default_rng(SEED + 80)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], check=False)
    for init in model.graph.initializer:
        if init.name == "L1_xzp":
            init.CopyFrom(numpy_helper.from_array(
                np.array(3, dtype=np.int8), name="L1_xzp"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="symmetric"):
        import_model(model, act)


def test_refuses_a_uint8_output_on_the_last_conv_layer():
    rng = np.random.default_rng(SEED + 81)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], check=False)
    for init in model.graph.initializer:
        if init.name == "L1_yzp":
            init.CopyFrom(numpy_helper.from_array(
                np.array(0, dtype=np.uint8), name="L1_yzp"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="int8 output"):
        import_model(model, act)


def test_refuses_a_single_conv_here_use_conv_matmul_chain_instead():
    # A single conv feeding matmuls routes to load_conv_matmul_chain
    # (import_model's own dispatch), not load_conv_chain_matmul_chain --
    # this just confirms that still works cleanly through the shared
    # dispatcher after narrowing its condition to conv_count == 1.
    from onnx_fixtures import conv_matmul_chain_model

    rng = np.random.default_rng(SEED + 82)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(w0, x_shape, [mm_w])
    import_model(model, act)  # does not raise
