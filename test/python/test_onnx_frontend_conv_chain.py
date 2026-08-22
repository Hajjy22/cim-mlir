"""A chain of two or more QLinearConv nodes, connected DIRECTLY -- the
real, hard capability docs/roadmap.md's "chain convolution layers" plan
set out to reach (PR C). See `cim_frontend.onnx_import.load_conv_chain`'s
own module section header for the full design, and
`cim_frontend.emit.emit_conv_chain_module`'s own docstring for why the
emitted IR looks the way it does (the interpreter's closed TypeSwitch,
the Kh*Kw-tap-unrolled static-stride gather, and the channel-last weight
flatten for every layer but the first).

WHY THIS NEEDS NO BRIDGE NODE, UNLIKE EVERY OTHER CHAIN THIS FRONT END
ACCEPTS
------------------------------------------------------------------------
Both QLinearConv's own "X" input and "Y" output are declared [N, C, H, W]
per the ONNX spec -- wiring one conv's "Y" directly into the next conv's
own "X" is already a faithful, standards-compliant statement of the same
function this compiler emits. Contrast with `conv_matmul_chain_model`'s
required Transpose/Reshape bridge (MatMulInteger has no native NCHW/NHWC
convention at all) and `matmul_chain_model`'s required Cast/QuantizeLinear
bridge (MatMulInteger's own "Y" has no in-node quantization of its own).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python"))

pytest.importorskip("onnx", reason="the ONNX front end's oracle")

from onnx import TensorProto  # noqa: E402
from onnx import helper as onnx_helper  # noqa: E402
from onnx import numpy_helper  # noqa: E402

from cim_frontend.onnx_import import import_model  # noqa: E402
from cim_frontend.refusal import Refusal  # noqa: E402
from onnx_fixtures import (conv_chain_model, onnx_reference_eval,  # noqa: E402
                          qlinear_conv_model)
from test_numerical_differential import compile_and_run, PipelineError  # noqa: E402

SEED = int(os.environ.get("CIM_TEST_SEED", "0x5eed1234abcd"), 0)


def _run(model, act, cim_opt, cim_run):
    """Returns (compiled_output, want), both flat and in the SAME row
    order.

    Unlike a chain ending in MatMulInteger (whose graph-declared output is
    already flat [M, N], the same order the compiled module prints),
    load_conv_chain's chains always end in QLinearConv -- ONNX declares
    that node's own "Y" as [N, Cout, OutH, OutW] (NCHW), while the
    compiled module keeps printing in (n, oh, ow)-row-major x Cout order
    throughout (emit_conv_chain_module threads M unchanged, exactly like
    emit_chain_module's own note) -- so `want` needs the same
    reshape-then-transpose(0, 2, 3, 1) im2col.py's own docstring already
    describes for a standalone conv (see also import_model's own
    conv-chain provenance header, which states this exact recipe)."""
    want = onnx_reference_eval(model, act, act_name="X")
    out_info = model.graph.output[0]
    n, cout, out_h, out_w = (d.dim_value for d in
                             out_info.type.tensor_type.shape.dim)
    want = want.reshape(n, cout, out_h, out_w).transpose(0, 2, 3, 1).ravel()

    text = import_model(model, act)
    try:
        outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                     source=text)
    except PipelineError as err:
        pytest.fail(f"the imported conv chain failed to compile or run: "
                    f"{err}")
    return np.asarray(outputs), want


# --- correctness -------------------------------------------------------

def test_a_two_layer_conv_chain_matches_the_reference(cim_opt, cim_run):
    rng = np.random.default_rng(SEED + 50)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_chain_model([w0, w1], x_shape)
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_conv_chain_with_stride_pad_and_dilation_together_matches_the_reference(
        cim_opt, cim_run):
    # The three parameters emit_conv_chain_module's own tap-gather has to
    # get right together on an INTERIOR layer, via static strides/offsets
    # rather than arithmetic (the interpreter cannot execute arith.addi/
    # muli -- see emit_conv_chain_module's own docstring): a strided,
    # padded, dilated gather from the previous layer's own bridged
    # [M, Cout] output.
    rng = np.random.default_rng(SEED + 51)
    w0 = rng.integers(-4, 5, size=(3, 2, 3, 3), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 7, 7)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_chain_model(
        [w0, w1], x_shape,
        strides=[(1, 1), (2, 2)], pads=[(0, 0, 0, 0), (1, 1, 1, 1)],
        dilations=[(1, 1), (2, 2)])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_batched_conv_chain_matches_the_reference(cim_opt, cim_run):
    # N > 1: M = N*OutH*OutW threads through the whole chain unchanged
    # (emit_chain_module's own note, reused unchanged by
    # emit_conv_chain_module), so a real batch exercises that the
    # expand_shape/padded-buffer/tap-gather geometry is derived from N
    # too, not just OutH*OutW.
    rng = np.random.default_rng(SEED + 52)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (2, 2, 4, 4)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_chain_model([w0, w1], x_shape)
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_three_layer_conv_chain_matches_the_reference(cim_opt, cim_run):
    # A chain longer than two layers -- exercises the channel-last weight
    # flatten (emit_conv_chain_module's own "dangerous line") on TWO
    # different interior/last layers, not just one, with distinct Cin/Cout
    # per layer so a wrong flatten order would not accidentally line up.
    rng = np.random.default_rng(SEED + 53)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(4, 3, 2, 2), dtype=np.int64).astype(np.int8)
    w2 = rng.integers(-4, 5, size=(2, 4, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_chain_model([w0, w1, w2], x_shape)
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_the_last_layers_bias_and_asymmetric_output_are_applied(
        cim_opt, cim_run):
    # Unlike a chain ending in MatMulInteger (PR B), QLinearConv's own "Y"
    # is DEFINED as the fully quantized result (bias, then requantize,
    # already applied) -- not a raw accumulator. This exercises the fix
    # this PR's own tail logic exists for: a real bias and an asymmetric
    # (uint8) y_zero_point on the LAST layer. y_scale stays 1.0 -- a REAL
    # (non-1.0) scale can land exactly on a rounding tie, where
    # cim.requantize's round-half-away-from-zero and ONNX QuantizeLinear's
    # round-half-to-even genuinely, correctly disagree by 1 (see
    # emit_chain_module's own docstring for the full argument); that is a
    # real hardware-fidelity fact, not a bug, but it is a DIFFERENT thing
    # to test than "is the bias/zero-point tail wired up at all", which is
    # this test's own job and is exact at scale=1.0 either way.
    rng = np.random.default_rng(SEED + 54)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    bias = np.array([3, -2], dtype=np.int32)

    model = conv_chain_model(
        [w0, w1], x_shape, last_bias=bias,
        last_y_zero_point=140, last_y_dtype=TensorProto.UINT8)
    outputs, want = _run(model, act, cim_opt, cim_run)
    # uint8_output_shifted: the compiled module's own printed values are
    # (true_uint8_value - 128) -- see load_qlinear_conv's own note, reused
    # unchanged for load_conv_chain's last layer.
    assert np.array_equal(outputs + 128, want), (
        f"compiled output + 128 = {(outputs + 128).tolist()} != ONNX "
        f"reference {want.tolist()}")


def test_a_wrong_weight_in_an_interior_layer_would_actually_be_caught(
        cim_opt, cim_run):
    # Anti-vacuity: a perturbation in the MIDDLE layer of a 3-layer chain
    # (neither first nor last) must still change the compiled result,
    # proving the chain actually threads every layer's own weight through
    # rather than the test only exercising the first/last layer.
    rng = np.random.default_rng(SEED + 55)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(4, 3, 2, 2), dtype=np.int64).astype(np.int8)
    w2 = rng.integers(-4, 5, size=(2, 4, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    base_model = conv_chain_model([w0, w1, w2], x_shape)
    base = onnx_reference_eval(base_model, act, act_name="X")

    bumped_w1 = w1.copy()
    bumped_w1[0, 0, 0, 0] = np.int8(int(bumped_w1[0, 0, 0, 0]) ^ 0x5A)
    bumped_model = conv_chain_model([w0, bumped_w1, w2], x_shape)
    unreshaped_want = onnx_reference_eval(bumped_model, act, act_name="X")
    assert not np.array_equal(base, unreshaped_want), (
        "the perturbation changed nothing")

    outputs, want = _run(bumped_model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want)


# --- Relu between two convs -------------------------------------------------

def test_a_relu_between_two_convs_matches_the_reference(cim_opt, cim_run):
    # w0/act chosen so layer 0's raw accumulator has both positive and
    # negative entries -- otherwise Relu is a no-op and this test would
    # pass vacuously (see the fixture-assumption assertion below).
    rng = np.random.default_rng(SEED + 58)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_chain_model([w0, w1], x_shape, relu_after={0})
    no_relu_model = conv_chain_model([w0, w1], x_shape)
    with_relu = onnx_reference_eval(model, act, act_name="X")
    without_relu = onnx_reference_eval(no_relu_model, act, act_name="X")
    assert not np.array_equal(with_relu, without_relu), (
        "fixture assumption broken: Relu made no difference for this "
        "weight/activation pair")

    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled Relu-between-convs output {outputs.tolist()} != ONNX "
        f"reference {want.tolist()}")


def test_a_missing_relu_between_convs_would_actually_be_caught(cim_opt,
                                                                cim_run):
    # Anti-vacuity: compiling WITH relu_after must match onnx.reference's
    # own Relu output, not the no-Relu one -- proving the compiled module
    # really emits the reduce_max step rather than it being present in
    # the IR but inert.
    rng = np.random.default_rng(SEED + 59)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_chain_model([w0, w1], x_shape, relu_after={0})
    outputs, want = _run(model, act, cim_opt, cim_run)
    no_relu_want = onnx_reference_eval(
        conv_chain_model([w0, w1], x_shape), act, act_name="X")
    assert not np.array_equal(want, no_relu_want)
    assert np.array_equal(outputs, want)


def test_refuses_a_relu_with_a_stray_extra_reader():
    # A Relu directly between two convs, but its OWN output is also read
    # by a stray extra node -- needs buffer-liveness reasoning this
    # importer does not do, exactly the matmul-chain side's own
    # equivalent refusal.
    rng = np.random.default_rng(SEED + 60)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)

    model = conv_chain_model([w0, w1], x_shape, relu_after={0}, check=False)
    # A second Relu reading the SAME tensor -- an allowed op TYPE, but a
    # fan-out _discover_conv_chain's own no-branching rule still refuses,
    # proving that rule is a real position/liveness check and not merely
    # a type allow-list.
    stray = onnx_helper.make_node(
        "Relu", ["L0_relu"], ["stray"], name="stray_relu")
    model.graph.node.append(stray)

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="2 total reader"):
        import_model(model, act)


def test_refuses_a_disconnected_relu_elsewhere_in_the_graph():
    # A Relu that is not on any edge _discover_conv_chain walks at all --
    # reads the graph's own input, feeds nothing -- exercises the
    # separate "every Relu actually present must be USED" check, not the
    # position-check the two tests above exercise.
    rng = np.random.default_rng(SEED + 61)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)

    model = conv_chain_model([w0, w1], x_shape, check=False)
    stray = onnx_helper.make_node("Relu", ["X"], ["stray"], name="stray_relu")
    model.graph.node.append(stray)

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="not positioned directly"):
        import_model(model, act)


# --- refusals ------------------------------------------------------------

def test_refuses_a_single_qlinear_conv():
    rng = np.random.default_rng(SEED + 56)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 3, 3)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    # A single conv is not a chain -- import_model routes it to
    # load_qlinear_conv instead, which happily accepts it (this just
    # confirms load_conv_chain never even sees the < 2 case in practice).
    model = qlinear_conv_model(w0, x_shape)
    import_model(model, act)  # does not raise


def test_refuses_a_stray_non_conv_node():
    # Relu is now accepted directly between two convs (see the tests
    # above) -- this checks a genuinely still-unsupported op type in
    # that exact position, so the "graph also contains ..." refusal is
    # proven to still fire for real, not just made vacuous by the Relu
    # acceptance.
    rng = np.random.default_rng(SEED + 57)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)

    model = conv_chain_model([w0, w1], x_shape, check=False)
    stray = onnx_helper.make_node("Sigmoid", ["L0_Y"], ["L0_Y_sig"], name="s0")
    model.graph.node.insert(1, stray)
    model.graph.node[2].input[0] = "L0_Y_sig"

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="Sigmoid"):
        import_model(model, act)


def test_refuses_a_branching_chain():
    # A conv whose output feeds TWO further convs is not a linear chain.
    rng = np.random.default_rng(SEED + 58)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)

    model = conv_chain_model([w0, w1], x_shape, check=False)
    extra = onnx_helper.make_node(
        "QLinearConv",
        ["L0_Y", "L1_xs", "L1_xzp", "L1_W", "L1_ws", "L1_wzp", "L1_ys",
         "L1_yzp"],
        ["L2_Y"], name="conv2", strides=[1, 1], pads=[0, 0, 0, 0])
    model.graph.node.append(extra)

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="not read as exactly one other"):
        import_model(model, act)


def test_refuses_a_disconnected_pair_of_convs():
    # Two convs that both read the graph's own input, neither reading the
    # other's output, is not a single chain (each is its own "start").
    rng = np.random.default_rng(SEED + 59)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)

    model = conv_chain_model([w0, w1], x_shape, check=False)
    model.graph.node[1].input[0] = "X"  # layer 1 reads X, not L0_Y

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="found 2 QLinearConv"):
        import_model(model, act)


def test_refuses_a_nonzero_y_zero_point_on_an_interior_layer():
    rng = np.random.default_rng(SEED + 60)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w2 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)

    model = conv_chain_model([w0, w1, w2], x_shape, check=False)
    for init in model.graph.initializer:
        if init.name == "L1_yzp":
            init.CopyFrom(numpy_helper.from_array(
                np.array(5, dtype=np.int8), name="L1_yzp"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="symmetric"):
        import_model(model, act)


def test_refuses_a_per_channel_w_scale_on_an_interior_layer():
    rng = np.random.default_rng(SEED + 61)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)

    model = conv_chain_model([w0, w1], x_shape, check=False)
    for init in model.graph.initializer:
        if init.name == "L0_ws":
            init.CopyFrom(numpy_helper.from_array(
                np.full(3, 1.0, dtype=np.float32), name="L0_ws"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="per-channel w_scale"):
        import_model(model, act)


def test_refuses_a_bias_on_an_interior_layer():
    rng = np.random.default_rng(SEED + 62)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)

    model = conv_chain_model([w0, w1], x_shape, check=False)
    for node in model.graph.node:
        if node.name == "conv0":
            model.graph.initializer.append(numpy_helper.from_array(
                np.zeros(2, dtype=np.int32), name="L0_B"))
            node.input.append("L0_B")

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="bias operand"):
        import_model(model, act)


def test_refuses_a_nonzero_x_zero_point_on_a_non_first_layer():
    rng = np.random.default_rng(SEED + 63)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)

    model = conv_chain_model([w0, w1], x_shape, check=False)
    for init in model.graph.initializer:
        if init.name == "L1_xzp":
            init.CopyFrom(numpy_helper.from_array(
                np.array(3, dtype=np.int8), name="L1_xzp"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="symmetric"):
        import_model(model, act)


def test_refuses_a_uint8_output_on_an_interior_layer():
    rng = np.random.default_rng(SEED + 64)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w2 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)

    model = conv_chain_model([w0, w1, w2], x_shape, check=False)
    for init in model.graph.initializer:
        if init.name == "L1_yzp":
            init.CopyFrom(numpy_helper.from_array(
                np.array(0, dtype=np.uint8), name="L1_yzp"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="int8 output"):
        import_model(model, act)


def test_refuses_a_cin_cout_mismatch_between_layers():
    rng = np.random.default_rng(SEED + 65)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 5, 5)

    model = conv_chain_model([w0, w1], x_shape, check=False)
    for init in model.graph.initializer:
        if init.name == "L1_W":
            bad = np.zeros((2, 4, 2, 2), dtype=np.int8)  # Cin=4, not 3
            init.CopyFrom(numpy_helper.from_array(bad, name="L1_W"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="own output channel count must match"):
        import_model(model, act)
