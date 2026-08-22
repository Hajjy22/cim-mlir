"""Pooling composed with the conv-to-matmul bridge -- the one gap
docs/roadmap.md's MaxPool plan explicitly deferred ("real, plausible
follow-on work, not attempted here"). Built by composing
`load_conv_pool_chain`'s own pool-aware conv-chain discovery with
`load_conv_chain_matmul_chain`'s own Transpose/Reshape bridge -- see
`cim_frontend.onnx_import.load_conv_pool_chain_matmul_chain`'s own module
section header for the full design.

THE ONE NEW ARITHMETIC CLAIM THIS FILE CHECKS
==================================================
Every other conv-chain-to-matmul test file inherits its M (the Reshape
bridge's own flattened row count) straight from the last conv layer's own
output size. Here, when a MaxPool sits between the last conv and the
bridge, M must reflect the POOLED spatial size instead -- a claim nothing
else in this front end exercises, since `load_conv_pool_chain` (interior
pooling only) has no bridge to compute M for at all, and
`load_conv_chain_matmul_chain` (the bridge, but no pooling) has no pool to
shrink M by. See this file's own mutation test.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python"))

pytest.importorskip("onnx", reason="the ONNX front end's oracle")

from onnx import helper as onnx_helper  # noqa: E402

from cim_frontend.onnx_import import import_model  # noqa: E402
from cim_frontend.refusal import Refusal  # noqa: E402
from onnx_fixtures import (conv_pool_chain_matmul_chain_model,  # noqa: E402
                          onnx_reference_eval)
from test_numerical_differential import compile_and_run, PipelineError  # noqa: E402

SEED = int(os.environ.get("CIM_TEST_SEED", "0x5eed1234abcd"), 0)


def _run(model, act, cim_opt, cim_run):
    """Exactly test_onnx_frontend_conv_chain_matmul_chain.py's own _run --
    the graph's own output is already flat [M, N] (a matmul chain's own
    raw accumulator), so unlike test_onnx_frontend_conv_pool_chain.py's
    own _run, no NCHW reshape/transpose is needed here regardless of
    whether pooling ran."""
    want = onnx_reference_eval(model, act, act_name="X")
    text = import_model(model, act)
    try:
        outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                     source=text)
    except PipelineError as err:
        pytest.fail(f"the imported conv-pool-chain-to-matmul chain failed "
                    f"to compile or run: {err}")
    return np.asarray(outputs), want


# --- correctness ---------------------------------------------------------

def test_a_trailing_pool_before_the_bridge_matches_the_reference(
        cim_opt, cim_run):
    # The new position: a MaxPool directly between the chain's own last
    # conv and the Transpose/Reshape bridge into a matmul.
    rng = np.random.default_rng(SEED + 70)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 9, 9)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={1: dict(kernel_shape=(2, 2))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_interior_and_trailing_pools_together_match_the_reference(
        cim_opt, cim_run):
    # Both positions in one graph: an interior pool (between conv0 and
    # conv1, load_conv_pool_chain's own established shape) AND a trailing
    # one (between conv1 and the bridge, the new shape), feeding TWO
    # matmul layers -- the fullest composition this loader supports.
    rng = np.random.default_rng(SEED + 71)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 13, 13)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w0 = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)
    mm_w1 = rng.integers(-4, 5, size=(4, 3), dtype=np.int64).astype(np.int8)

    # bridge_scales must be ODD -- an even inter-matmul scale can hit a
    # genuine round-half-to-even/round-half-away-from-zero tie (python/
    # README.md's own "Rounding" section; this was confirmed the hard way
    # while developing this test: 2.0 produced 8 rows that "failed" only
    # because -33/2 = -16.5 is a real tie, not a bug in this loader).
    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w0, mm_w1],
        pools={0: dict(kernel_shape=(2, 2)), 1: dict(kernel_shape=(2, 2))},
        bridge_scales=[3.0])
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_relu_composed_with_an_interior_pool_matches_the_reference(
        cim_opt, cim_run):
    # Relu directly before the interior pool (Conv -> Relu -> MaxPool ->
    # Conv), the same two-intermediate-node shape load_conv_pool_chain's
    # own test already proves, here reused unchanged by this loader's
    # own matmul tail.
    rng = np.random.default_rng(SEED + 86)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 9, 9)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={0: dict(kernel_shape=(2, 2))}, relu_after={0})
    no_relu_model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], pools={0: dict(kernel_shape=(2, 2))})
    with_relu = onnx_reference_eval(model, act, act_name="X")
    without_relu = onnx_reference_eval(no_relu_model, act, act_name="X")
    assert not np.array_equal(with_relu, without_relu), (
        "fixture assumption broken: Relu made no difference")

    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_relu_directly_before_the_trailing_pool_matches_the_reference(
        cim_opt, cim_run):
    # Conv -> Relu -> MaxPool -> Transpose: the one bridge position this
    # loader's own module header used to name as unrecognized. No
    # emitter change needed -- every bridge emit_conv_chain_module emits
    # already has zero_point == 0, the only thing Relu's own
    # reduce_max(x, 0) composition needs -- this is pure discovery-side
    # wiring, the same "one extra position past what the interior walk
    # covers" pattern used for the trailing pool itself.
    rng = np.random.default_rng(SEED + 99)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 9, 9)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={1: dict(kernel_shape=(2, 2))}, relu_after={1})
    no_relu_model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], pools={1: dict(kernel_shape=(2, 2))})
    with_relu = onnx_reference_eval(model, act, act_name="X")
    without_relu = onnx_reference_eval(no_relu_model, act, act_name="X")
    assert not np.array_equal(with_relu, without_relu), (
        "fixture assumption broken: Relu made no difference")

    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_relu_directly_before_the_matmul_bridge_with_no_trailing_pool_matches_the_reference(
        cim_opt, cim_run):
    # The same bridge position, but with no pool sitting there (the
    # interior pool at bridge 0 satisfies this loader's own "at least
    # one MaxPool in the graph" requirement) -- proves the trailing
    # Relu-hop works whether or not a pool follows it at the SAME
    # bridge, exactly load_conv_chain_matmul_chain's own unpooled
    # acceptance.
    rng = np.random.default_rng(SEED + 99)
    w0 = rng.integers(-4, 5, size=(3, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 3, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 9, 9)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 4), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={0: dict(kernel_shape=(2, 2))}, relu_after={1})
    no_relu_model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], pools={0: dict(kernel_shape=(2, 2))})
    with_relu = onnx_reference_eval(model, act, act_name="X")
    without_relu = onnx_reference_eval(no_relu_model, act, act_name="X")
    assert not np.array_equal(with_relu, without_relu), (
        "fixture assumption broken: Relu made no difference")

    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_refuses_a_relu_before_the_trailing_pool_with_a_stray_extra_reader():
    # A second reader of the trailing Relu's own output -- the "must
    # feed exactly one reader" guard the trailing Relu-hop adds,
    # mirroring load_conv_chain_matmul_chain's own identical bridge
    # fan-out check.
    rng = np.random.default_rng(SEED + 100)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 8, 8)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={1: dict(kernel_shape=(2, 2))}, relu_after={1}, check=False)
    stray = onnx_helper.make_node(
        "Relu", ["L1_relu"], ["stray_out"], name="stray_relu_reader")
    model.graph.node.append(stray)

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="must feed exactly one reader"):
        import_model(model, act)


def test_refuses_a_disconnected_relu_elsewhere_when_a_trailing_relu_is_present():
    # A Relu that is not on any edge this loader (or
    # _discover_conv_pool_chain, which it calls) ever walks, present
    # alongside a LEGITIMATE trailing Relu in the same graph.
    rng = np.random.default_rng(SEED + 101)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 8, 8)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={1: dict(kernel_shape=(2, 2))}, relu_after={1}, check=False)
    stray = onnx_helper.make_node("Relu", ["X"], ["stray"], name="stray_relu")
    model.graph.node.append(stray)

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="not positioned directly"):
        import_model(model, act)


def test_a_padded_trailing_pool_matches_the_reference(cim_opt, cim_run):
    # Real, asymmetric padding on the trailing pool -- exercises the -128
    # fill at the one new bridge position, against a negative-heavy
    # activation where an incorrect pad value would win a max it should
    # lose (test_onnx_frontend_conv_pool_chain.py's own precedent).
    rng = np.random.default_rng(SEED + 72)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 8, 8)
    act = rng.integers(-40, -20, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={1: dict(kernel_shape=(3, 3), strides=(2, 2),
                      pads=(1, 1, 1, 1))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_batched_trailing_pool_chain_matches_the_reference(
        cim_opt, cim_run):
    rng = np.random.default_rng(SEED + 73)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (2, 2, 9, 9)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={1: dict(kernel_shape=(2, 2))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want), (
        f"compiled output {outputs.tolist()} != ONNX reference "
        f"{want.tolist()}")


def test_a_wrong_matmul_weight_after_the_trailing_pool_would_be_caught(
        cim_opt, cim_run):
    # Anti-vacuity: a perturbation in the matmul layer that reads the
    # POOLED activation must still change the compiled result, proving the
    # pool's own output is actually threaded into the bridge rather than
    # this test only exercising the conv portion.
    rng = np.random.default_rng(SEED + 74)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 8, 8)
    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w], pools={1: dict(kernel_shape=(2, 2))})
    outputs, want = _run(model, act, cim_opt, cim_run)
    assert np.array_equal(outputs, want)

    bad_mm_w = mm_w.copy()
    bad_mm_w[0, 0] += 1
    bad_model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [bad_mm_w], pools={1: dict(kernel_shape=(2, 2))})
    bad_outputs, _ = _run(bad_model, act, cim_opt, cim_run)
    assert not np.array_equal(bad_outputs, outputs), (
        "perturbing the matmul weight after the trailing pool did not "
        "change the compiled output -- the pool's own result is not "
        "actually reaching the matmul layer")


# --- refusals --------------------------------------------------------------

def test_refuses_a_trailing_pool_with_stride_one():
    # Same oracle gap as every other MaxPool position in this front end
    # (onnx.reference cannot evaluate an integer MaxPool at stride 1 at
    # all) -- this loader's own trailing-pool detection must apply
    # _pool_geometry's existing check, not skip it.
    rng = np.random.default_rng(SEED + 75)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={1: dict(kernel_shape=(2, 2), strides=(1, 1))})

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="cannot evaluate"):
        import_model(model, act)


def test_refuses_a_missing_bridge_after_a_trailing_pool():
    # The pool must still be followed by the real Transpose/Reshape
    # bridge -- a trailing pool does not exempt the graph from needing
    # one, it just moves the bridge's own source.
    rng = np.random.default_rng(SEED + 76)
    w0 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-4, 5, size=(2, 2, 2, 2), dtype=np.int64).astype(np.int8)
    x_shape = (1, 2, 6, 6)
    mm_w = rng.integers(-4, 5, size=(2, 3), dtype=np.int64).astype(np.int8)

    model = conv_pool_chain_matmul_chain_model(
        [w0, w1], x_shape, [mm_w],
        pools={1: dict(kernel_shape=(2, 2))})
    # Sever the bridge: point the first matmul directly at the pool's own
    # output instead of the Reshape's.
    for node in model.graph.node:
        if node.name == "mm0":
            del node.input[0]
            node.input.insert(0, "L1_pool")

    act = rng.integers(-4, 5, size=x_shape, dtype=np.int64).astype(np.int8)
    with pytest.raises(Refusal, match="Transpose"):
        import_model(model, act)


def test_an_unpooled_conv_chain_matmul_chain_still_routes_correctly(
        cim_opt, cim_run):
    # Dispatch regression guard, the direction that actually matters: this
    # loader's new dispatch branch sits BEFORE load_conv_chain_matmul_
    # chain's own in import_model, guarded on pool_count >= 1. If that
    # guard were dropped or loosened, a plain (pool-free) conv-chain-to-
    # matmul graph would be wrongly routed into
    # load_conv_pool_chain_matmul_chain instead -- which would still work
    # here (pool_params would just be empty), so this is a routing check,
    # not a correctness one: confirms the existing, pool-free composition
    # still produces a correct answer with the new branch sitting in front
    # of it.
    from onnx_fixtures import conv_chain_matmul_chain_model
    rng = np.random.default_rng(SEED + 77)
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
