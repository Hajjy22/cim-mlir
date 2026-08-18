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
(onnx_import.py) for what is deliberately still refused (a non-zero
w_zero_point, non-explicit padding) and why each is a real scope boundary
rather than an oversight. Dilation is accepted (see im2col.py's own
"DILATION IS A SAMPLING PATTERN" note) -- it changes which source pixels a
kernel tap reads, not the shape of anything. Grouped (and depthwise)
convolution is accepted too, but NOT as a reshape: it decomposes into
`group` independent matmuls instead -- see
emit.emit_grouped_conv_module's own docstring for why, and this file's
own grouped/depthwise tests below.

PER-CHANNEL SCALE, BIAS, AND ASYMMETRIC ZERO POINTS ARE ALSO RESHAPES
------------------------------------------------------------------------
A real quantized model (squeezenet1.0-12-int8, pulled from the ONNX model
zoo while researching what to build next) uses per-channel `w_scale`, a
real int32 bias, and an asymmetric `x_zero_point` on every single layer --
none of which the first version of this front end accepted. All three
turned out to be expressible with existing, already-verified machinery
(N `cim.requantize` calls over column subviews for per-channel scale, a
`cim.reduce_partial` for the bias, a Python-side constant shift for
x_zero_point), not a dialect change -- see load_qlinear_conv's own module
section header for the full reasoning and the real round trips that
confirmed each one before this file's own tests were written.

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

import onnx  # noqa: E402

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


def test_a_grouped_conv_matches_the_quantized_reference(cim_opt, cim_run):
    # group=2, Cin=4, Cout=6 -- a REAL grouped conv (Cin/group=2, not the
    # degenerate Cin/group=1 depthwise case below), so this exercises the
    # general "group's own weight slice keeps its own Cin/group input
    # channels" shape, not just the special case. See emit.
    # emit_grouped_conv_module's own docstring for why this decomposes
    # into `group` independent matmuls rather than one padded dense one.
    rng = np.random.default_rng(SEED + 24)
    cout, cin_per_group, kh, kw, group = 6, 2, 2, 2, 2
    weight = rng.integers(-6, 7, size=(cout, cin_per_group, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin_per_group * group, 5, 5)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=5.0, group=group)
    want = onnx_reference_eval(model, act, act_name="X")
    assert np.abs(want).max() < 100

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=4, out_w=4)
    assert np.array_equal(got, want.reshape(1, cout, 4, 4)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()}")


def test_a_depthwise_conv_matches_the_quantized_reference(cim_opt, cim_run):
    # group == Cin == Cout: the degenerate, most common real case
    # (MobileNet-style networks) -- each output channel reads exactly one
    # input channel, Cin/group == 1.
    rng = np.random.default_rng(SEED + 25)
    channels, kh, kw = 4, 3, 3
    weight = rng.integers(-6, 7, size=(channels, 1, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, channels, 6, 6)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=5.0, group=channels)
    want = onnx_reference_eval(model, act, act_name="X")
    assert np.abs(want).max() < 100

    got = _run_conv(cim_opt, cim_run, model, act, channels, out_h=4, out_w=4)
    assert np.array_equal(got, want.reshape(1, channels, 4, 4)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()}")


def test_a_batched_grouped_conv_matches_the_quantized_reference(cim_opt,
                                                                 cim_run):
    # M > 1 batching composes with grouping the same way it already does
    # with im2col's own batch flattening (test_a_batched_conv_matches_
    # the_quantized_reference above) -- a combination neither feature's
    # own tests exercise on its own.
    rng = np.random.default_rng(SEED + 26)
    cout, cin_per_group, kh, kw, group = 4, 2, 2, 2, 2
    weight = rng.integers(-5, 6, size=(cout, cin_per_group, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (2, cin_per_group * group, 4, 4)
    act = rng.integers(-5, 6, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=5.0, group=group)
    want = onnx_reference_eval(model, act, act_name="X")
    assert np.abs(want).max() < 100

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3, n=2)
    assert np.array_equal(got, want.reshape(2, cout, 3, 3)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()}")


def test_a_wrong_weight_in_one_group_would_actually_be_caught(cim_opt,
                                                               cim_run):
    # Anti-vacuity: a perturbation confined to group 1's own weight must
    # change the compiled result -- proving group 1's own matmul (not just
    # group 0's) is actually threaded into the concatenated output, not
    # silently dropped or overwritten by the other group's own subview
    # copy.
    rng = np.random.default_rng(SEED + 27)
    cout, cin_per_group, kh, kw, group = 4, 2, 2, 2, 2
    weight = rng.integers(-6, 7, size=(cout, cin_per_group, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin_per_group * group, 5, 5)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=5.0, group=group)
    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=4, out_w=4)

    bad_weight = weight.copy()
    cout_per_group = cout // group
    bad_weight[cout_per_group] += 1  # group 1's own first output channel
    bad_model = qlinear_conv_model(bad_weight, x_shape, x_scale=1.0,
                                   w_scale=1.0, y_scale=5.0, group=group)
    bad_got = _run_conv(cim_opt, cim_run, bad_model, act, cout, out_h=4,
                        out_w=4)
    assert not np.array_equal(bad_got, got), (
        "perturbing group 1's own weight did not change the compiled "
        "output -- its matmul is not actually reaching the concatenated "
        "result")


def test_refuses_a_group_that_does_not_evenly_divide_cout():
    rng = np.random.default_rng(SEED + 28)
    weight = rng.integers(-4, 5, size=(3, 2, 2, 2),
                          dtype=np.int64).astype(np.int8)  # Cout=3, group=2
    x_shape = (1, 4, 4, 4)
    act = np.zeros(x_shape, dtype=np.int8)

    model = qlinear_conv_model(weight, x_shape, group=2)
    with pytest.raises(Refusal, match="group"):
        import_model(model, act)


def test_a_dilated_conv_matches_the_quantized_reference(cim_opt, cim_run):
    # dilations=(2, 2), asymmetric stride (2, 1) so a swapped H/W axis
    # anywhere in the dilation plumbing -- onnx_import.py's own
    # dilation_h/dilation_w unpacking, or im2col_nchw's -- reads as a
    # shape mismatch or a wrong number, not a lucky match. Kernel/input
    # sized so the dilated receptive field (3x3 effective, from a 2x2
    # kernel at dilation 2) fits a 5x5 input with room for a real,
    # non-trivial output grid.
    rng = np.random.default_rng(SEED + 26)
    cout, cin, kh, kw = 2, 2, 2, 2
    weight = rng.integers(-6, 7, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 5, 5)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=5.0, strides=(2, 1),
                               dilations=(2, 2))
    want = onnx_reference_eval(model, act, act_name="X")
    assert np.abs(want).max() < 100, (
        "this fixture is meant to stay far from the +-128 clamp; a "
        "saturated result here would mean a future edit to the seed or "
        "weight range quietly defeated the point of this test")

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=2, out_w=3)
    assert np.array_equal(got, want.reshape(1, cout, 2, 3)), (
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


def test_a_conv_with_a_bias_matches_the_quantized_reference(cim_opt, cim_run):
    # A real int32 bias, per output channel, distinct values (50, -30, 10)
    # so a row/column mix-up in the cim.reduce_partial broadcast reads as
    # a loud, wrong number rather than a suspiciously-plausible one.
    rng = np.random.default_rng(SEED + 40)
    cout, cin = 3, 2
    weight = rng.integers(-6, 7, size=(cout, cin, 2, 2),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 4, 4)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)
    bias = np.array([50, -30, 10], dtype=np.int32)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=3.0, bias=bias)
    want = onnx_reference_eval(model, act, act_name="X")
    assert np.abs(want).max() < 100

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3)
    assert np.array_equal(got, want.reshape(1, cout, 3, 3)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()}")


def test_a_per_channel_scale_conv_matches_the_quantized_reference(cim_opt,
                                                                   cim_run):
    # Three deliberately different w_scale values (1.0, 3.0, 7.0) -- a
    # bug that applied one channel's scale (or an average) to every
    # channel would compute plausible-looking but wrong numbers for at
    # least two of the three, not a shape error.
    rng = np.random.default_rng(SEED + 41)
    cout, cin = 3, 2
    weight = rng.integers(-3, 4, size=(cout, cin, 2, 2),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 4, 4)
    act = rng.integers(-3, 4, size=x_shape, dtype=np.int64).astype(np.int8)
    w_scale = np.array([1.0, 3.0, 7.0], dtype=np.float32)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=w_scale,
                               y_scale=1.0)
    want = onnx_reference_eval(model, act, act_name="X")
    assert np.abs(want).max() < 127

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3)
    assert np.array_equal(got, want.reshape(1, cout, 3, 3)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()}")


def test_an_asymmetric_x_zero_point_matches_the_quantized_reference(cim_opt,
                                                                     cim_run):
    # X declared uint8 with x_zero_point=120 -- the real, common
    # convention this front end's own investigation found in a real
    # quantized model. Activation values cluster near 120 (uint8
    # [115, 125]) so the shifted, signed result stays small and
    # non-saturating, isolating the zero-point-subtraction correctness
    # question from the earlier saturation-masks-everything lesson.
    rng = np.random.default_rng(SEED + 42)
    cout, cin = 2, 2
    weight = rng.integers(-5, 6, size=(cout, cin, 2, 2),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 4, 4)
    act = rng.integers(115, 126, size=x_shape, dtype=np.int64).astype(np.uint8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=3.0, x_zero_point=120,
                               x_dtype=onnx.TensorProto.UINT8)
    want = onnx_reference_eval(model, act, act_name="X",
                               activation_dtype=np.uint8)
    assert np.abs(want).max() < 100

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3)
    assert np.array_equal(got, want.reshape(1, cout, 3, 3)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()}")


def test_an_asymmetric_y_zero_point_matches_the_quantized_reference(cim_opt,
                                                                     cim_run):
    # A real (non-zero, negative) output zero point -- proves
    # cim.requantize's own zero_point attribute (already exercised at a
    # different value in test/mlir/legalize_precision_e2e_test.cpp) is
    # correctly threaded through from a real ONNX model's y_zero_point,
    # not just hard-coded to 0 as the first version of this front end did.
    rng = np.random.default_rng(SEED + 43)
    cout, cin = 2, 2
    weight = rng.integers(-6, 7, size=(cout, cin, 2, 2),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 4, 4)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=3.0, y_zero_point=-20)
    want = onnx_reference_eval(model, act, act_name="X")
    assert np.abs(want).max() < 100

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3)
    assert np.array_equal(got, want.reshape(1, cout, 3, 3)), (
        f"compiled {got.tolist()} != ONNX's own quantized reference "
        f"{want.tolist()}")


def test_a_uint8_output_matches_the_quantized_reference_after_the_shift(
    cim_opt, cim_run
):
    # squeezenet1.0-12-int8's real first layer declares a uint8
    # y_zero_point (docs/roadmap.md's own M4 entry names this as the one
    # gap the capstone verification exposed) -- cim.requantize's clamp is
    # SIGNED, so load_qlinear_conv requantizes with (y_zero_point - 128)
    # instead and the emitted module's own provenance header discloses
    # that every printed value is (true_uint8_value - 128), exactly (see
    # onnx_import.py's own algebraic proof: clamp(-128,127,t-128) ==
    # clamp(0,255,t)-128 for every real t). This test is that proof
    # exercised on real compiled output: `want` is onnx.reference's own
    # TRUE uint8 values; `got + 128` is what the compiled pipeline
    # produces, asserted equal.
    #
    # y_zero_point=130 -- comfortably nonzero and not itself 128, so a
    # bug that forgot the shift entirely (comparing `got` to `want`
    # directly) or applied a DIFFERENT constant (off-by-one, or
    # subtracting 127) would not accidentally still pass.
    rng = np.random.default_rng(SEED + 44)
    cout, cin = 2, 2
    weight = rng.integers(-6, 7, size=(cout, cin, 2, 2),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 4, 4)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)

    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=3.0, y_zero_point=130,
                               y_dtype=onnx.TensorProto.UINT8)
    want = onnx_reference_eval(model, act, act_name="X")
    assert want.dtype == np.uint8
    assert 0 < want.min() and want.max() < 255, (
        "this fixture is meant to stay off both ends of uint8's range; a "
        "result pinned to 0 or 255 everywhere would make the shift "
        "invisible regardless of whether it is correct")

    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3)
    shifted_back = got.astype(np.int64) + 128
    assert np.array_equal(shifted_back, want.astype(np.int64).reshape(1, cout, 3, 3)), (
        f"compiled {got.tolist()} + 128 != ONNX's own quantized reference "
        f"{want.tolist()}")


# --- refusals ------------------------------------------------------------

def _base_model(cout=2, cin=2, kh=2, kw=2, h=4, w=4, **kwargs):
    rng = np.random.default_rng(SEED + 30)
    weight = rng.integers(-10, 11, size=(cout, cin, kh, kw),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, h, w)
    return weight, x_shape, qlinear_conv_model(weight, x_shape, **kwargs)


def test_refuses_a_non_positive_dilation():
    # A real dilation (>= 1) is accepted -- see
    # test_a_dilated_conv_matches_the_quantized_reference below -- but 0
    # or negative is not a sampling pattern this front end (or a real
    # accelerator) can express. Set through qlinear_conv_model's own
    # `dilations` kwarg, not by appending a second `dilations` attribute
    # onto the node by hand: the fixture always emits ITS OWN `dilations`
    # attribute (default (1, 1)), so appending a second one would leave
    # the node with two, and onnx_import.py's `_int_list_attr` reads the
    # FIRST match -- silently validating the fixture's own (1, 1), not
    # this test's (0, 1), and never refusing at all.
    weight, x_shape, model = _base_model(dilations=(0, 1))
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="dilations"):
        import_model(model, act)


def test_refuses_a_non_zero_w_zero_point():
    # The one zero point still refused -- see load_qlinear_conv's own
    # module section header on why w_zero_point specifically does not
    # reduce to an existing reshape the way x_zero_point/y_zero_point do.
    weight, x_shape, model = _base_model()
    for init in model.graph.initializer:
        if init.name == "w_zp":
            model.graph.initializer.remove(init)
            break
    model.graph.initializer.append(onnx.numpy_helper.from_array(
        np.array(1, dtype=np.int8), name="w_zp"))
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="symmetric"):
        import_model(model, act)


def test_refuses_a_non_zero_per_channel_w_zero_point():
    # Same restriction, but at the per-channel shape a real quantization
    # tool actually emits -- see test_accepts_an_all_zero_per_channel_
    # w_zero_point below for why the all-zero case of this exact shape
    # must NOT be refused (found via a real model: squeezenet1.0-12-int8's
    # own first layer ships a length-64, all-zero w_zero_point, not a
    # true scalar).
    weight, x_shape, model = _base_model()
    cout = weight.shape[0]
    for init in model.graph.initializer:
        if init.name == "w_zp":
            model.graph.initializer.remove(init)
            break
    per_channel = np.zeros(cout, dtype=np.int8)
    per_channel[-1] = 1
    model.graph.initializer.append(
        onnx.numpy_helper.from_array(per_channel, name="w_zp"))
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="symmetric"):
        import_model(model, act)


def test_accepts_an_all_zero_per_channel_w_zero_point(cim_opt, cim_run):
    # "prove it doesn't refuse everything" -- found while validating this
    # front end against a real model (squeezenet1.0-12-int8's first
    # layer): w_zero_point is commonly a length-Cout array even when
    # every entry is trivially 0, not always a true scalar the way
    # x_zero_point/y_zero_point are. All-zero is still symmetric
    # regardless of shape, so it must be accepted.
    rng = np.random.default_rng(SEED + 44)
    cout, cin = 3, 2
    weight = rng.integers(-6, 7, size=(cout, cin, 2, 2),
                          dtype=np.int64).astype(np.int8)
    x_shape = (1, cin, 4, 4)
    act = rng.integers(-6, 7, size=x_shape, dtype=np.int64).astype(np.int8)
    model = qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                               y_scale=3.0)
    for init in model.graph.initializer:
        if init.name == "w_zp":
            model.graph.initializer.remove(init)
            break
    model.graph.initializer.append(onnx.numpy_helper.from_array(
        np.zeros(cout, dtype=np.int8), name="w_zp"))
    onnx.checker.check_model(model)

    want = onnx_reference_eval(model, act, act_name="X")
    got = _run_conv(cim_opt, cim_run, model, act, cout, out_h=3, out_w=3)
    assert np.array_equal(got, want.reshape(1, cout, 3, 3))


def test_refuses_an_x_zero_point_that_overflows_int8_after_shift():
    # x=0 (uint8) minus x_zero_point=200 is -200 -- outside signed int8's
    # [-128, 127] no matter what w_zero_point does, so this must be
    # refused rather than silently wrapped.
    weight, x_shape, model = _base_model(
        x_zero_point=200, x_dtype=onnx.TensorProto.UINT8)
    act = np.zeros(x_shape, dtype=np.uint8)
    with pytest.raises(Refusal, match="does not fit signed int8"):
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
    # This duplicates the SAME node (same output name, same activation
    # input) rather than building a real chain, so import_model's
    # conv_count >= 2 dispatch now routes it to load_conv_chain (PR C),
    # not load_qlinear_conv -- and load_conv_chain refuses it for its own,
    # equally accurate reason: two conv nodes whose own activation is the
    # graph's own input is not a single linear chain (see its own
    # docstring). Before load_conv_chain existed, ANY two-or-more-conv
    # graph fell through to load_qlinear_conv's "only a single standalone
    # convolution" refusal instead -- this test's match string tracks
    # that change, not just the plain "still refused" fact, so it does not
    # accidentally silently start validating the wrong loader's message.
    weight, x_shape, model = _base_model()
    extra = model.graph.node[0]
    duplicated = model.graph.node.add()
    duplicated.CopyFrom(extra)
    duplicated.name = "conv2"
    act = np.zeros(x_shape, dtype=np.int8)
    with pytest.raises(Refusal, match="own activation is not another"):
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


def test_im2col_with_dilation_matches_a_hand_written_convolution():
    """Same independence goal as the test above, but with a real dilation
    != 1 -- see im2col.py's own 'DILATION IS A SAMPLING PATTERN' note.
    The oracle here deliberately does NOT use numpy's strided-slice
    syntax (`start:stop:step`) the way im2col_nchw's own implementation
    does -- reusing that trick in the "independent" check would really be
    testing the same formula against itself. Instead it visits each
    (kh_i, kw_i) tap one at a time and computes its source pixel by hand
    (`h0 + kh_i * dilation_h`), which is the literal definition of a
    dilated convolution rather than a reimplementation of im2col's own
    indexing. Deliberately asymmetric (dilation_h != dilation_w,
    stride_h != stride_w, pad != 0) so a swapped H/W axis anywhere in the
    dilation plumbing reads as a shape error or a wrong number, not a
    lucky match."""
    rng = np.random.default_rng(SEED + 25)
    n, c, h, w = 2, 3, 9, 7
    cout, kh, kw = 4, 3, 2
    stride_h, stride_w = 2, 1
    dilation_h, dilation_w = 3, 2
    pad = (1, 0, 1, 0)  # top, bottom, left, right

    x = rng.integers(-10, 11, size=(n, c, h, w))
    weight = rng.integers(-10, 11, size=(cout, c, kh, kw))

    patches, (out_n, out_h, out_w) = im2col_nchw(
        x, kh, kw, stride_h, stride_w, *pad,
        dilation_h=dilation_h, dilation_w=dilation_w)
    got = (patches @ weight.reshape(cout, -1).T).reshape(
        out_n, out_h, out_w, cout).transpose(0, 3, 1, 2)

    xp = np.pad(x, ((0, 0), (0, 0), (pad[0], pad[1]), (pad[2], pad[3])))
    want = np.zeros((n, cout, out_h, out_w), dtype=np.int64)
    for ni in range(n):
        for co in range(cout):
            for oh in range(out_h):
                for ow in range(out_w):
                    h0, w0 = oh * stride_h, ow * stride_w
                    acc = 0
                    for ci in range(c):
                        for khi in range(kh):
                            for kwi in range(kw):
                                pixel = xp[ni, ci,
                                          h0 + khi * dilation_h,
                                          w0 + kwi * dilation_w]
                                acc += int(pixel) * int(weight[co, ci, khi, kwi])
                    want[ni, co, oh, ow] = acc

    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.array_equal(got, want)
