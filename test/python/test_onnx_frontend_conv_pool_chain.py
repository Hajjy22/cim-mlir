"""A chain of two or more QLinearConv nodes with a MaxPool optionally
sitting between any two consecutive layers -- the first non-matmul host op
this front end executes (docs/roadmap.md's MaxPool plan, PR B). See
`cim_frontend.onnx_import.load_conv_pool_chain`'s own module section
header for the full design, and `cim_frontend.emit.emit_conv_chain_module`'s
own `pool_params` docstring for why the emitted IR looks the way it does
(cim.reduce_max folding Kh*Kw strided taps directly, no per-tap slab
copy).

WHY THIS FILE ALSO CARRIES ITS OWN, HAND-WRITTEN MAXPOOL ORACLE
==================================================================
Every other differential test in this front end trusts `onnx.reference`
alone as the primary oracle. MaxPool is the one place that trust does not
hold: `onnx.reference` cannot evaluate an INTEGER MaxPool at strides == 1
at all -- it pads with `np.nan` as its own "never wins the max" sentinel,
which `np.pad` refuses to place into an integer array, and raises
`ValueError: cannot convert float NaN to integer` for every kernel/shape/
pads combination at stride 1 (confirmed by hand against the installed
package before this feature was designed; see docs/roadmap.md's MaxPool
entry). This front end sidesteps that gap by refusing strides == 1
outright (see `test_refuses_a_pool_with_stride_one` below) -- but AT the
strides this front end does accept (>= 2), resting correctness on
`onnx.reference` alone would still be unwise, given it is demonstrably
buggy in this exact area. `_independent_maxpool_nchw` below is a second,
by-hand oracle -- deliberately not built from `im2col.py`'s own
`output_size()`/`_pad_nchw()` (reusing those would really be testing the
pipeline against itself), matching
`test_onnx_frontend_conv.py`'s own "independent hand-written convolution"
precedent for exactly the same reason.
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

from cim_frontend.onnx_import import import_model  # noqa: E402
from cim_frontend.refusal import Refusal  # noqa: E402
from onnx_fixtures import (conv_chain_model, conv_pool_chain_model,  # noqa: E402
                          onnx_reference_eval)
from test_numerical_differential import compile_and_run, PipelineError  # noqa: E402

SEED = int(os.environ.get("CIM_TEST_SEED", "0x5eed1234abcd"), 0)


def _independent_maxpool_nchw(x, kh, kw, stride_h, stride_w,
                              pad_top, pad_bottom, pad_left, pad_right,
                              dilation_h=1, dilation_w=1):
    """A direct, by-hand [N, C, H, W] MaxPool loop -- see this file's own
    module docstring for why it deliberately does not call im2col.py's own
    `output_size()`/`_pad_nchw()`. Pads with -128 (INT8_MIN) directly: the
    literal statement of what a signed-int8 MaxPool's implicit padding
    means -- a padded position never wins a max against a real one.

    `dilation_h`/`dilation_w` (default 1, ordinary pooling) space the taps
    inside each window `dilation` elements apart via a strided slice --
    same idea as `im2col.py`'s own dilation support, deliberately
    reimplemented here rather than shared, for the same "don't test the
    pipeline against itself" reason the whole function exists.
    """
    x = np.asarray(x, dtype=np.int64)
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom),
                    (pad_left, pad_right)),
               mode="constant", constant_values=-128)
    hp, wp = xp.shape[2], xp.shape[3]
    dilated_kh = (kh - 1) * dilation_h + 1
    dilated_kw = (kw - 1) * dilation_w + 1
    out_h = (hp - dilated_kh) // stride_h + 1
    out_w = (wp - dilated_kw) // stride_w + 1
    out = np.empty((n, c, out_h, out_w), dtype=np.int64)
    for oh in range(out_h):
        h0 = oh * stride_h
        for ow in range(out_w):
            w0 = ow * stride_w
            window = xp[:, :, h0:h0 + dilated_kh:dilation_h,
                       w0:w0 + dilated_kw:dilation_w]
            out[:, :, oh, ow] = window.max(axis=(2, 3))
    return out


def _run(model, act, cim_opt, cim_run):
    """Returns (compiled_output, want), both flat and in the SAME row
    order -- exactly `test_onnx_frontend_conv_chain.py`'s own `_run`
    helper (see its docstring for the reshape-then-transpose(0, 2, 3, 1)
    recipe; a MaxPool between two conv layers does not change it, since
    `emit_conv_chain_module` keeps printing in (n, oh, ow)-row-major x
    Cout order throughout regardless of whether pooling ran)."""
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
        pytest.fail(f"the imported conv/pool chain failed to compile or "
                    f"run: {err}")
    return np.asarray(outputs), want


# --- the independent oracle, checked against onnx.reference itself -------

def test_the_independent_maxpool_oracle_matches_onnx_reference():
    # Trust, but verify the verifier: before using
    # _independent_maxpool_nchw to check this front end's OWN output, this
    # confirms it agrees with onnx.reference at strides >= 2 (the only
    # regime the primary oracle can evaluate at all for an integer
    # MaxPool -- see this file's own module docstring), including real
    # padding.
    rng = np.random.default_rng(SEED + 70)
    x = rng.integers(-100, 100, size=(2, 3, 7, 7)).astype(np.int64)
    kh, kw, stride_h, stride_w = 3, 3, 2, 2
    pad = (1, 1, 1, 1)  # top, bottom, left, right

    got = _independent_maxpool_nchw(x, kh, kw, stride_h, stride_w, *pad)

    model = onnx_helper.make_model(
        onnx_helper.make_graph(
            [onnx_helper.make_node(
                "MaxPool", ["X"], ["Y"], kernel_shape=[kh, kw],
                strides=[stride_h, stride_w],
                pads=[pad[0], pad[2], pad[1], pad[3]])],
            "pool_only",
            [onnx_helper.make_tensor_value_info(
                "X", TensorProto.INT8, list(x.shape))],
            [onnx_helper.make_tensor_value_info(
                "Y", TensorProto.INT8, list(got.shape))]),
        opset_imports=[onnx_helper.make_opsetid("", 13)])
    want = onnx_reference_eval(model, x.astype(np.int8), act_name="X")
    assert np.array_equal(got.ravel(), want), (
        "the independent oracle disagrees with onnx.reference at a "
        "stride the reference CAN evaluate -- the oracle itself is "
        "suspect, not (necessarily) the front end")


def test_the_independent_maxpool_oracle_matches_onnx_reference_with_dilation():
    # Same self-check, at real dilation (2, 2) plus real padding -- proving
    # the independent oracle's own dilation support (added alongside
    # MaxPool dilation acceptance in the front end) agrees with
    # onnx.reference before it is trusted to check this front end's output
    # below. A deliberately asymmetric kernel/dilation/stride combination,
    # not a repeated (2, 2) everywhere, so a swapped h/w axis in either
    # implementation could not accidentally cancel out.
    rng = np.random.default_rng(SEED + 83)
    x = rng.integers(-100, 100, size=(2, 3, 9, 8)).astype(np.int64)
    kh, kw, stride_h, stride_w = 2, 3, 2, 2
    dilation_h, dilation_w = 2, 1
    pad = (1, 0, 0, 1)  # top, bottom, left, right

    got = _independent_maxpool_nchw(x, kh, kw, stride_h, stride_w, *pad,
                                    dilation_h=dilation_h,
                                    dilation_w=dilation_w)

    model = onnx_helper.make_model(
        onnx_helper.make_graph(
            [onnx_helper.make_node(
                "MaxPool", ["X"], ["Y"], kernel_shape=[kh, kw],
                strides=[stride_h, stride_w],
                dilations=[dilation_h, dilation_w],
                pads=[pad[0], pad[2], pad[1], pad[3]])],
            "pool_only",
            [onnx_helper.make_tensor_value_info(
                "X", TensorProto.INT8, list(x.shape))],
            [onnx_helper.make_tensor_value_info(
                "Y", TensorProto.INT8, list(got.shape))]),
        opset_imports=[onnx_helper.make_opsetid("", 13)])
    want = onnx_reference_eval(model, x.astype(np.int8), act_name="X")
    assert np.array_equal(got.ravel(), want), (
        "the independent oracle's own dilation support disagrees with "
        "onnx.reference -- the oracle itself is suspect, not "
        "(necessarily) the front end")


# --- correctness, through the real cim-opt/cim-run pipeline ---------------

def test_a_conv_pool_conv_chain_matches_the_reference(cim_opt, cim_run):
    rng = np.random.default_rng(SEED + 71)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(4, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 8, 8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_model(
        [w0, w1], x_shape, pools={0: dict(kernel_shape=(2, 2))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_padded_pool_matches_the_reference(cim_opt, cim_run):
    # Real, asymmetric padding on the pool -- exercises the -128 fill
    # against a genuinely negative-heavy activation, where an incorrect
    # (e.g. 0) pad value would win a max it should lose (see this file's
    # own mutation-test companion, which flips this exact value).
    rng = np.random.default_rng(SEED + 72)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 7, 7)
    act = rng.integers(-40, -20, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_model(
        [w0, w1], x_shape,
        pools={0: dict(kernel_shape=(3, 3), strides=(2, 2),
                      pads=(1, 1, 1, 1))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_overlapping_pool_windows_match_the_reference(cim_opt, cim_run):
    # kernel > stride: adjacent pooling windows share taps -- a different
    # gather shape than the non-overlapping case above (the same static
    # subview machinery, but with real overlap between taps).
    rng = np.random.default_rng(SEED + 73)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 9, 9)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_model(
        [w0, w1], x_shape,
        pools={0: dict(kernel_shape=(3, 3), strides=(2, 2))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_dilated_pool_matches_the_reference(cim_opt, cim_run):
    # dilations != (1, 1) on the pool itself -- accepted, not refused
    # (onnx_import.py's own module header explains why this is not an
    # oracle gap the way stride == 1 is). Real, asymmetric padding too, so
    # the -128 fill is exercised at the same time as the dilated tap
    # pattern, not as two separate claims.
    rng = np.random.default_rng(SEED + 84)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 9, 9)
    act = rng.integers(-40, -20, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_model(
        [w0, w1], x_shape,
        pools={0: dict(kernel_shape=(2, 2), strides=(2, 2),
                      pads=(1, 1, 0, 0), dilations=(2, 2))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_the_full_conv_pool_conv_pool_conv_shape_matches_the_reference(
        cim_opt, cim_run):
    # The realistic early-layer shape of a small classification CNN --
    # test/python/onnx_fixtures.py's own small_multi_layer_cnn_model
    # already anticipated this shape for cost analysis; this is the first
    # test that actually EXECUTES it.
    rng = np.random.default_rng(SEED + 74)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(4, 3, 2, 2), dtype=np.int64).astype(np.int8)
    w2 = rng.integers(-4, 5, size=(2, 4, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 12, 12)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_model(
        [w0, w1, w2], x_shape,
        pools={0: dict(kernel_shape=(2, 2)), 1: dict(kernel_shape=(2, 2))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_batched_conv_pool_chain_matches_the_reference(cim_opt, cim_run):
    rng = np.random.default_rng(SEED + 75)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (2, 2, 6, 6)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_model(
        [w0, w1], x_shape, pools={0: dict(kernel_shape=(2, 2))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_wrong_weight_after_the_pool_would_actually_be_caught(
        cim_opt, cim_run):
    # Anti-vacuity: a perturbation in the LAST conv layer -- the one that
    # reads the POOLED activation -- must still change the compiled
    # result, proving the pool's own output is actually threaded into the
    # next layer's gather rather than the test only exercising layer 0.
    rng = np.random.default_rng(SEED + 76)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 8, 8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    base_model = conv_pool_chain_model(
        [w0, w1], x_shape, pools={0: dict(kernel_shape=(2, 2))})
    base = onnx_reference_eval(base_model, act, act_name="X")

    bumped_w1 = w1.copy()
    bumped_w1[0, 0, 0, 0] = np.int8(int(bumped_w1[0, 0, 0, 0]) ^ 0x5A)
    bumped_model = conv_pool_chain_model(
        [w0, bumped_w1], x_shape, pools={0: dict(kernel_shape=(2, 2))})
    bumped_want = onnx_reference_eval(bumped_model, act, act_name="X")
    assert not np.array_equal(base, bumped_want), (
        "the perturbation changed nothing")

    outputs, want = _run(bumped_model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want)


# --- refusals --------------------------------------------------------------

def test_refuses_a_pool_with_stride_one():
    rng = np.random.default_rng(SEED + 77)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)

    model = conv_pool_chain_model(
        [w0, w1], x_shape,
        pools={0: dict(kernel_shape=(2, 2), strides=(1, 1))})

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="cannot evaluate"):
        import_model(model, act)


def test_refuses_a_pool_with_non_positive_dilation():
    # dilations != (1, 1) is now accepted (see test_a_dilated_pool_matches_
    # the_reference above) -- but a non-positive entry is still refused,
    # the same "not a sampling pattern any real accelerator can express"
    # rule _conv_geometry already applies to a convolution's own dilation.
    rng = np.random.default_rng(SEED + 85)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 8, 8)

    model = conv_pool_chain_model(
        [w0, w1], x_shape,
        pools={0: dict(kernel_shape=(2, 2), dilations=(0, 1))})

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="dilations"):
        import_model(model, act)


def test_refuses_ceil_mode():
    rng = np.random.default_rng(SEED + 78)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 7, 7)

    model = conv_pool_chain_model(
        [w0, w1], x_shape,
        pools={0: dict(kernel_shape=(2, 2), ceil_mode=1)})

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="ceil_mode"):
        import_model(model, act)


def test_refuses_a_non_notset_auto_pad():
    rng = np.random.default_rng(SEED + 79)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)

    model = conv_pool_chain_model(
        [w0, w1], x_shape, pools={0: dict(kernel_shape=(2, 2))})
    for node in model.graph.node:
        if node.name == "pool0":
            node.attribute.append(
                onnx_helper.make_attribute("auto_pad", "SAME_UPPER"))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="auto_pad"):
        import_model(model, act)


def test_refuses_a_pool_after_the_chains_own_last_layer():
    # A pool at the chain's own edge (after the last conv, feeding the
    # graph's own output directly) is outside load_conv_pool_chain's own
    # scope -- see its module header's "WHAT IS STILL REFUSED" note.
    rng = np.random.default_rng(SEED + 80)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)

    model = conv_pool_chain_model(
        [w0, w1], x_shape, pools={}, check=False)
    tail_pool = onnx_helper.make_node(
        "MaxPool", ["L1_Y"], ["Pooled"], name="pool_tail",
        kernel_shape=[2, 2], strides=[2, 2])
    model.graph.node.append(tail_pool)
    del model.graph.output[:]
    model.graph.output.append(onnx_helper.make_tensor_value_info(
        "Pooled", TensorProto.INT8, [1, 2, 2, 2]))

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal,
                       match="not positioned directly between two "
                             "consecutive"):
        import_model(model, act)


def test_refuses_a_stray_non_conv_non_pool_node():
    rng = np.random.default_rng(SEED + 81)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)

    model = conv_pool_chain_model(
        [w0, w1], x_shape, pools={0: dict(kernel_shape=(2, 2))}, check=False)
    stray = onnx_helper.make_node("Relu", ["L0_pool"], ["L0_pool_relu"],
                                  name="r0")
    model.graph.node.insert(2, stray)
    model.graph.node[3].input[0] = "L0_pool_relu"

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="Relu"):
        import_model(model, act)


def test_pure_conv_chain_with_no_pool_still_routes_to_load_conv_chain():
    # A regression guard for the dispatch change itself: a graph with NO
    # MaxPool node must still take load_conv_chain's own path (PR C),
    # unaffected by this PR's new branch in import_model.
    rng = np.random.default_rng(SEED + 82)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 4, 4)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)

    model = conv_chain_model([w0, w1], x_shape)
    import_model(model, act)  # does not raise
