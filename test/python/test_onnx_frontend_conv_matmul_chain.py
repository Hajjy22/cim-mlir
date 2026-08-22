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
from onnx_fixtures import (conv_chain_matmul_chain_model,  # noqa: E402
                           conv_matmul_chain_model, onnx_reference_eval)
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


# --- grouped/depthwise conv as the chain's own layer 0 ----------------------

def test_a_grouped_conv_feeding_a_matmul_matches_the_reference(cim_opt,
                                                                cim_run):
    # Grouped conv IS accepted here: load_conv_matmul_chain's own conv
    # layer is ALWAYS the chain's own layer 0, and
    # emit_grouped_conv_matmul_chain_module produces the same flat
    # [M, Cout] activation an ungrouped conv's own emit_chain_module
    # bridge already expects -- see load_conv_matmul_chain's own
    # docstring and _conv_geometry's allow_grouped note for why this is
    # different from every OTHER chain loader, which still refuses.
    cout, cin_per_group, kh, kw, group = 4, 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 39)
    conv_w = rng.integers(-4, 5, size=(cout, cin_per_group, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin_per_group * group, 5, 5)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], group=group)
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled grouped-conv-matmul-chain output {outputs.tolist()} != "
        f"ONNX reference {want.tolist()}")


def test_a_depthwise_conv_feeding_a_matmul_matches_the_reference(cim_opt,
                                                                  cim_run):
    # group == Cin == Cout: the fully depthwise special case.
    cin_out, kh, kw, group = 4, 2, 2, 4
    rng = np.random.default_rng(SEED + 40)
    conv_w = rng.integers(-4, 5, size=(cin_out, 1, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin_out, 5, 5)
    mm_w = rng.integers(-4, 5, size=(cin_out, 3), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], group=group)
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled depthwise-conv-matmul-chain output {outputs.tolist()} "
        f"!= ONNX reference {want.tolist()}")


def test_a_wrong_weight_in_one_group_of_a_conv_matmul_chain_would_be_caught(
        cim_opt, cim_run):
    # Anti-vacuity: perturbing group 1's own conv weight (not group 0's,
    # and not the matmul's) must still change the compiled result --
    # proving the concatenated groups actually feed the matmul, not just
    # group 0's slice.
    cout, cin_per_group, kh, kw, group = 4, 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 41)
    conv_w = rng.integers(-4, 5, size=(cout, cin_per_group, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin_per_group * group, 5, 5)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    base_model = conv_matmul_chain_model(conv_w, x_shape, [mm_w], group=group)
    base = onnx_reference_eval(base_model, act, act_name="X")

    cout_per_group = cout // group
    bumped_w = conv_w.copy()
    bumped_w[cout_per_group, 0, 0, 0] = np.int8(
        int(bumped_w[cout_per_group, 0, 0, 0]) ^ 0x5A)
    bumped_model = conv_matmul_chain_model(bumped_w, x_shape, [mm_w],
                                           group=group)
    want = onnx_reference_eval(bumped_model, act, act_name="X")
    assert not np.array_equal(base, want), "the perturbation changed nothing"

    outputs, _ = compile_and_run(
        cim_opt, cim_run, None, None,
        source=import_model(bumped_model, act))
    assert np.array_equal(np.asarray(outputs), want)


def test_refuses_a_grouped_conv_in_a_two_conv_chain():
    # Unlike load_conv_matmul_chain (single conv, always layer 0),
    # load_conv_chain_matmul_chain's own conv layers gather each other's
    # bridged output via emit_conv_chain_module's 4-D NHWC machinery,
    # built for one dense channel count -- a grouped layer's own
    # per-group Cin/Cout has not been threaded through that gather, so
    # this loader's own _conv_geometry call keeps its allow_grouped=False
    # default. Confirms the acceptance above is genuinely scoped to
    # load_conv_matmul_chain, not a blanket lift.
    cout0, cin0, kh, kw = 4, 2, 2, 2
    cout1 = 3
    rng = np.random.default_rng(SEED + 42)
    conv_w0 = rng.integers(-4, 5, size=(cout0, cin0, kh, kw),
                           dtype=np.int64).astype(np.int8)
    conv_w1 = rng.integers(-4, 5, size=(cout1, cout0, kh, kw),
                           dtype=np.int64).astype(np.int8)
    x_shape = (1, cin0, 6, 6)
    mm_w = rng.integers(-4, 5, size=(cout1, 3), dtype=np.int64).astype(np.int8)

    model = conv_chain_matmul_chain_model([conv_w0, conv_w1], x_shape, [mm_w])
    for node in model.graph.node:
        if node.op_type == "QLinearConv" and node.name == "conv0":
            node.attribute.append(onnx_helper.make_attribute("group", 2))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="group"):
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


def _rename_tensor(model, old, new):
    """Rename one initializer and every reference to it, in place."""
    for init in model.graph.initializer:
        if init.name == old:
            init.name = new
    for node in model.graph.node:
        node.input[:] = [new if i == old else i for i in node.input]
    return model


def test_a_grouped_chain_with_two_matmul_layers_and_a_real_scale(cim_opt,
                                                                  cim_run):
    # The grouped-conv chain's MULTI-matmul-layer path: the `scales`
    # parameter and the intermediate cim.requantize bridge between matmul
    # layers. Every other grouped-chain test passes a single matmul
    # layer, which never reaches that code at all -- a real coverage hole
    # this closes. The bridge scale is deliberately non-1.0 (and odd, so
    # a rounding tie is impossible -- see test_onnx_frontend_chain.py's
    # own note) to prove the scale is threaded through rather than
    # silently hardcoded.
    cout, cin_per_group, kh, kw, group = 4, 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 43)
    conv_w = rng.integers(-4, 5, size=(cout, cin_per_group, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin_per_group * group, 5, 5)
    mm_w0 = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)
    mm_w1 = rng.integers(-4, 5, size=(3, 2), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_matmul_chain_model(conv_w, x_shape, [mm_w0, mm_w1],
                                    group=group, bridge_scales=[3.0])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled two-matmul grouped chain {outputs.tolist()} != ONNX "
        f"reference {want.tolist()}")


def test_an_onnx_weight_named_like_an_internal_ssa_value_still_compiles(
        cim_opt, cim_run):
    # REGRESSION: the emitters used to name SSA values after the ONNX
    # tensor (`%{sym} = memref.get_global @{sym}`), so a model whose
    # matmul weight happened to be called "w0" -- the same name the
    # grouped conv's own group-0 weight uses internally -- emitted
    # `%w0` twice and cim-opt rejected the module with "redefinition of
    # SSA value". A loud failure rather than a wrong number, but a
    # perfectly ordinary ONNX name should not break the front end at
    # all. SSA names are now generated positionally and independently of
    # any ONNX name (matching emit_module, which always did this).
    cout, cin_per_group, kh, kw, group = 4, 2, 2, 2, 2
    rng = np.random.default_rng(SEED + 44)
    conv_w = rng.integers(-4, 5, size=(cout, cin_per_group, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin_per_group * group, 5, 5)
    mm_w = rng.integers(-4, 5, size=(cout, 3), dtype=np.int64).astype(np.int8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = _rename_tensor(
        conv_matmul_chain_model(conv_w, x_shape, [mm_w], group=group),
        "mmW0", "w0")
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want)


def test_the_grouped_chain_emitter_refuses_duplicate_symbols():
    # The three symbol lists share ONE module-level namespace (each name
    # becomes a memref.global). import_model can never violate this --
    # it runs every name through sanitize_symbol against one shared
    # `taken` set -- but a direct caller could, and used to get an opaque
    # "redefinition of symbol" out of cim-opt instead of a clear error.
    from cim_frontend.emit import emit_grouped_conv_matmul_chain_module

    with pytest.raises(ValueError, match="distinct"):
        emit_grouped_conv_matmul_chain_module(
            [np.array([[1, 0]], np.int8), np.array([[0, 1]], np.int8)],
            [np.array([3, 4], np.int8), np.array([5, 6], np.int8)],
            [np.array([[1, 1]], np.int8)],
            mm_weight_syms=["w0"])   # collides with the conv's default w0
