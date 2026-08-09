"""A single 2-D convolution: the front end's second entry point.

WHY A CONVOLUTION IS A MATMUL HERE, WHEN IT ISN'T IN GENERAL
--------------------------------------------------------------
onnx_import.py's own module docstring establishes that every emitted
module is ONE BAKED INFERENCE: cim-run has no mechanism to pass data in,
so the activation is always a compile-time constant. That single fact is
what makes im2col free here, when it is usually considered an expensive
trick: there is no per-inference cost to worry about, because there is no
per-inference cost at all in this project's execution model. `QLinearConv`
is reshaped into the SAME (weight[N, K], activation[M, K]) shape a plain
MatMulInteger already produces -- see im2col.py's own module docstring for
the exact reshape, and load_qlinear_conv's module section header
(onnx_import.py) for what is deliberately still refused (grouped
convolution, dilation, per-channel quantization, a bias operand,
non-explicit padding) and why each is a real scope boundary rather than an
oversight.

THE SCALE ARITHMETIC IS DIFFERENT FROM THE CHAIN BRIDGE'S
------------------------------------------------------------
QLinearConv, unlike MatMulInteger, has no "raw int32" form to import: it
always quantizes its own output in the same node, via `x_scale`,
`w_scale`, and `y_scale`. `load_qlinear_conv` derives the single scalar
`cim.requantize` needs from those three:
`derived_scale = y_scale / (x_scale * w_scale)`, so that
`round(res / derived_scale) == round(res * (x_scale * w_scale / y_scale))`
-- QLinearConv's own reference formula -- for `res` an integer (raw,
pre-scale) accumulator. The tests below pick `x_scale = w_scale = 1.0`
throughout, so `derived_scale == y_scale` directly and the choice of
`y_scale` can be reasoned about the same way test_onnx_frontend_chain.py
reasons about a bridge's scale: cim.requantize rounds half-away-from-zero,
QuantizeLinear (which QLinearConv's reference implementation also uses)
rounds half-to-even, and those two modes diverge only at a tie.

THE OUTPUT LAYOUT BOOKKEEPING
--------------------------------
The compiled pipeline only ever emits a flat matmul result, [M, Cout] with
M = N*OutH*OutW in (n, oh, ow) row-major order (im2col.py's own docstring
names this order explicitly) -- reshaping that back to ONNX's own
[N, Cout, OutH, OutW] is pure bookkeeping this test file does once, in
`_nhwc_flat_to_nchw`, rather than something the importer does (the
importer's job ends at "produce the right numbers"; a caller wanting
ONNX's own axis order applies this reshape itself, exactly as
import_model's own emitted provenance comment says to).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python"))

pytest.importorskip("onnx", reason="the ONNX front end's oracle")

from cim_frontend.im2col import im2col_nchw  # noqa: E402
from cim_frontend.onnx_import import import_model  # noqa: E402
from cim_frontend.refusal import Refusal  # noqa: E402
from onnx_fixtures import onnx_reference_eval, qlinear_conv_model  # noqa: E402
from test_numerical_differential import compile_and_run, PipelineError  # noqa: E402

SEED = int(os.environ.get("CIM_TEST_SEED", "0x5eed1234abcd"), 0)


def _nhwc_flat_to_nchw(flat, n, out_h, out_w, cout):
    """The compiled pipeline's own flat [M, Cout] print, reshaped to ONNX's
    [N, Cout, OutH, OutW] -- see this file's own module docstring."""
    return (np.asarray(flat).reshape(n, out_h, out_w, cout)
           .transpose(0, 3, 1, 2))


def _run_conv(cim_opt, cim_run, model, act, cout, out_h, out_w, n=1):
    text = import_model(model, act)
    try:
        outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                     source=text)
    except PipelineError as err:
        pytest.fail(f"the imported convolution failed to compile or run: "
                    f"{err}")
    return _nhwc_flat_to_nchw(outputs, n, out_h, out_w, cout)


# --- correctness -------------------------------------------------------

def test_a_strided_padded_conv_matches_the_quantized_reference(cim_opt,
                                                                cim_run):
    # Small (not full-int8-range) weights and activations, and a modest
    # y_scale=5 -- both deliberately chosen so the raw accumulator stays
    # well inside [-128, 127] and NOTHING here saturates. That choice is
    # load-bearing, not cosmetic: test_onnx_frontend_chain.py's own
    # mutation-testing history found that full-range weights make a
    # requantize's OWN scale irrelevant, because every accumulator
    # saturates to the same +-128 clamp regardless -- a test that passes
    # for that reason proves nothing about whether the scale threads
    # through correctly. Checked by hand for this seed: raw accumulator
    # magnitudes here stay under 20, an order of magnitude inside the
    # clamp.
    rng = np.random.default_rng(SEED + 20)
    cout, cin, kh, kw = 3, 2, 2, 2
    weight = rng.integers(-6, 7, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 5, 5)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=5.0, strides=(2, 2),
                               pads=(1, 1, 1, 1))
    want = onnx_reference_eval(model, act, act_name="X")
    assert np.abs(want).max() < 100, (
        "this fixture is meant to stay far from the +-128 clamp; a "
        "saturated result here would mean a future edit to the seed or "
        "weight range quietly defeated the point of this test")

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3)
    assert np.array_equal(got, want.reshape(1, cout, 3, 3)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()}")


def test_a_batched_conv_matches_the_quantized_reference(cim_opt, cim_run):
    # N=3 images flattened into the matmul's M dimension -- proves the
    # M > 1 batching machinery (docs/roadmap.md's M4 entry) composes with
    # im2col's own batch flattening, a combination neither feature's own
    # tests exercise on its own. y_scale=5 (odd), not 4: an even scale
    # found a real, expected rounding-tie divergence in exactly the
    # sense test_onnx_frontend_chain.py's own even-scale test documents
    # (cim.requantize rounds half-away-from-zero, QuantizeLinear rounds
    # half-to-even) -- caught here while first writing this test, not a
    # hypothetical concern.
    rng = np.random.default_rng(SEED + 21)
    cout, cin, kh, kw = 2, 2, 3, 3
    weight = rng.integers(-5, 6, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (3, cin, 4, 4)
    act = rng.integers(-5, 6, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=5.0)
    want = onnx_reference_eval(model, act, act_name="X")
    assert np.abs(want).max() < 100

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=2, out_w=2, n=3)
    assert np.array_equal(got, want.reshape(3, cout, 2, 2)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()}")


def test_a_1x1_conv_at_an_odd_scale_matches_exactly(cim_opt, cim_run):
    # A 1x1 kernel over full-int8-range data (K = Cin = 4, small enough
    # that the accumulator's worst case, 4*127*127, still needs only an
    # odd y_scale of 509 or more to guarantee no rounding tie -- the same
    # v = scale*(n + 0.5) argument test_onnx_frontend_chain.py's own odd-
    # scale test relies on: scale/2 is never an integer when scale is
    # odd, so no integer accumulator ever lands exactly on a tie). This is
    # the convolution path's own version of that guarantee, independent
    # of the "pick a small, non-saturating scale by hand" case above.
    rng = np.random.default_rng(SEED + 22)
    cout, cin = 3, 4
    weight = rng.integers(-128, 128, size=(cout, cin, 1, 1),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 3, 3)
    act = rng.integers(-128, 128, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=509.0)
    want = onnx_reference_eval(model, act, act_name="X")

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3)
    assert np.array_equal(got, want.reshape(1, cout, 3, 3)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()} -- an odd scale should make this an exact "
        f"match, never a rounding-tie mismatch")


def test_a_conv_at_an_exact_rounding_tie_documents_the_known_divergence(
        cim_opt, cim_run):
    # Deliberately hand-constructed, not random -- the convolution's own
    # version of test_onnx_frontend_chain.py's identically-named matmul
    # test. A single 1x1 kernel over a single pixel: raw accumulator
    # 1*1 = 1, y_scale=2.0 (even) makes 1/2 = 0.5 an exact tie.
    # cim.requantize rounds half-away-from-zero (-> 1); QuantizeLinear
    # (which QLinearConv's own reference implementation also uses) rounds
    # half-to-even (-> 0). Both are correct for their own documented
    # rounding mode; this test exists to name the divergence explicitly
    # rather than let it surface as an unexplained near-miss, exactly the
    # discipline test_a_batched_conv_matches_the_quantized_reference's own
    # comment above points back to.
    weight = np.array([[[[1]]]], dtype=np.int8)  # Cout=1, Cin=1, Kh=1, Kw=1
    x_shape = (1, 1, 1, 1)
    act = np.array([[[[1]]]], dtype=np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=2.0)
    onnx_result = onnx_reference_eval(model, act, act_name="X")
    got = _run_conv(cim_opt, cim_run, model, act, cout=1, out_h=1, out_w=1)

    assert onnx_result.tolist() == [0], (
        "the oracle's own rounding at this tie changed; re-derive this "
        "test's constants")
    assert got.reshape(-1).tolist() == [1], (
        "cim.requantize's rounding at this tie changed; re-derive this "
        "test's constants")
    assert int(got.reshape(-1)[0]) - int(onnx_result[0]) == 1, (
        "the two rounding modes should disagree by exactly 1 at this "
        "hand-picked tie")


# --- refusals ------------------------------------------------------------

def _base_model(cout=2, cin=2, kh=2, kw=2, h=4, w=4, **kwargs):
    rng = np.random.default_rng(SEED + 30)
    weight = rng.integers(-10, 11, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, h, w)
    return weight, x_shape, qlinear_conv_model(weight, x_shape, **kwargs)


def test_refuses_a_grouped_convolution():
    weight, x_shape, model = _base_model()
    model.graph.node[0].attribute.append(
        __import__("onnx").helper.make_attribute("group", 2))
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="group"):
        import_model(model, act)


def test_refuses_a_dilated_convolution():
    weight, x_shape, model = _base_model()
    model.graph.node[0].attribute.append(
        __import__("onnx").helper.make_attribute("dilations", [2, 2]))
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="dilation"):
        import_model(model, act)


def test_refuses_a_bias_operand():
    import onnx
    weight, x_shape, model = _base_model()
    cout = weight.shape[0]
    bias = np.zeros(cout, dtype=np.int32)
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(bias, name="B"))
    model.graph.node[0].input.append("B")
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="bias"):
        import_model(model, act)


def test_refuses_a_non_scalar_w_scale():
    import onnx
    weight, x_shape, model = _base_model()
    cout = weight.shape[0]
    for init in model.graph.initializer:
        if init.name == "w_scale":
            model.graph.initializer.remove(init)
            break
    model.graph.initializer.append(onnx.numpy_helper.from_array(
        np.ones(cout, dtype=np.float32), name="w_scale"))
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="per-output-channel"):
        import_model(model, act)


@pytest.mark.parametrize("which", ["x_zp", "w_zp", "y_zp"])
def test_refuses_a_non_zero_zero_point(which):
    import onnx
    weight, x_shape, model = _base_model()
    for init in model.graph.initializer:
        if init.name == which:
            model.graph.initializer.remove(init)
            break
    model.graph.initializer.append(onnx.numpy_helper.from_array(
        np.array(1, dtype=np.int8), name=which))
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="symmetric"):
        import_model(model, act)


def test_refuses_a_non_positive_scale():
    weight, x_shape, model = _base_model()
    for init in model.graph.initializer:
        if init.name == "y_scale":
            init.raw_data = np.array(-1.0, dtype=np.float32).tobytes()
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="positive"):
        import_model(model, act)


@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
def test_refuses_non_notset_auto_pad(auto_pad):
    # VALID is refused too, alongside SAME_UPPER/SAME_LOWER, despite being
    # definitionally just pads=[0, 0, 0, 0] -- see load_qlinear_conv's own
    # "WHAT IS DELIBERATELY REFUSED" note: onnx.reference's own Conv
    # implementation computes VALID's padding with buggy, shape-dependent
    # arithmetic (indexing X.shape[i] for a spatial dim without skipping
    # the N/C axes), so it cannot be used as an oracle for it. Found while
    # writing test_a_conv_with_no_padding_matches_the_quantized_reference
    # below -- not a hypothetical concern.
    weight, x_shape, model = _base_model(auto_pad=auto_pad)
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="auto_pad"):
        import_model(model, act)


def test_refuses_a_second_qlinear_conv_node():
    weight, x_shape, model = _base_model()
    extra = model.graph.node[0]
    duplicated = model.graph.node.add()
    duplicated.CopyFrom(extra)
    duplicated.name = "conv2"
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="only a single standalone"):
        import_model(model, act)


def test_refuses_a_uint8_weight():
    weight, x_shape, model = _base_model()
    for init in model.graph.initializer:
        if init.name == "W":
            init.data_type = 2  # TensorProto.UINT8, same bytes reinterpreted
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="dtype"):
        import_model(model, act)


def test_a_conv_with_no_padding_matches_the_quantized_reference(cim_opt,
                                                                 cim_run):
    # "prove it doesn't refuse everything" -- NOTSET with an explicit
    # pads=[0, 0, 0, 0] (also this front end's own default when `pads` is
    # entirely absent) is accepted and gets the exact answer VALID would,
    # without relying on onnx.reference's own buggy VALID/SAME_* padding
    # computation as an oracle (see test_refuses_non_notset_auto_pad).
    # Also exercises im2col_nchw's early-return, no-padding-needed branch,
    # which the strided/padded test above never reaches.
    rng = np.random.default_rng(SEED + 23)
    cout, cin = 2, 2
    weight = rng.integers(-6, 7, size=(cout, cin, 2, 2),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 4, 4)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)
    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=3.0, pads=(0, 0, 0, 0))
    want = onnx_reference_eval(model, act, act_name="X")
    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3)
    assert np.array_equal(got, want.reshape(1, cout, 3, 3))


# --- im2col.py: pure numpy, no onnx dependency ----------------------------

def test_im2col_matches_a_hand_written_convolution():
    """im2col_nchw needs no onnx installed to test -- see its own module
    docstring -- so this checks it against a direct, independent (not
    im2col-based) numpy convolution loop rather than against onnx at all."""
    rng = np.random.default_rng(SEED + 24)
    n, c, h, w = 2, 3, 5, 5
    cout, kh, kw = 4, 3, 3
    stride_h, stride_w = 2, 1
    pad = (1, 0, 1, 0)  # top, bottom, left, right

    x = rng.integers(-10, 11, size=(n, c, h, w))
    weight = rng.integers(-10, 11, size=(cout, c, kh, kw))

    patches, (out_n, out_h, out_w) = im2col_nchw(
        x, kh, kw, stride_h, stride_w, *pad)
    got = (patches @ weight.reshape(cout, -1).T).reshape(
        out_n, out_h, out_w, cout).transpose(0, 3, 1, 2)

    xp = np.pad(x, ((0, 0), (0, 0), (pad[0], pad[1]), (pad[2], pad[3])))
    want = np.zeros((n, cout, out_h, out_w), dtype=np.int64)
    for ni in range(n):
        for co in range(cout):
            for oh in range(out_h):
                for ow in range(out_w):
                    h0, w0 = oh * stride_h, ow * stride_w
                    window = xp[ni, :, h0:h0 + kh, w0:w0 + kw]
                    want[ni, co, oh, ow] = int(
                        np.sum(window.astype(np.int64) *
                              weight[co].astype(np.int64)))

    assert np.array_equal(got, want)
