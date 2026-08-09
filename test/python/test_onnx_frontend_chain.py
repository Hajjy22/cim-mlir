"""Chained layers: the multi-layer counterpart to test_onnx_frontend.py.

A single MatMulInteger closes docs/roadmap.md's M2 box, but it is also the
one shape where cim-placement has nothing to reuse ACROSS: one weight
matrix, used once, is exactly the case
placement_never_changes_the_numbers_on_any_single_matmul_case already
covers. A chain of layers -- multiple weight matrices competing for a
fixed tile budget -- is what the mlp-3layer benchmark shape actually is,
and it is the first time that shape is reachable from a real model file
rather than a generated one.

THE BRIDGE, AND WHY IT IS SAFE AGAINST AN UNQUANTIZED ORACLE
--------------------------------------------------------------
Layer i's int32 accumulator has to become layer i+1's int8 activation.
cim_frontend accepts exactly one way to do that:
Cast(to=float32) -> QuantizeLinear(scale=1.0, zero_point=0, int8 out),
which it lowers to a real cim.requantize(scale=1.0, zero_point=0,
effective_bits=8) sitting between the two matmuls in the emitted MLIR.

scale=1.0 is not a simplification, it is the reason this stays exact:
QuantizeLinear rounds half-to-even and cim.requantize rounds
half-away-from-zero, and those two modes diverge only at a tie -- a value
ending in exactly .5. With scale=1.0 both sides compute round(v / 1.0) on
a v that is already an integer (an accumulator), so there is never a
fractional part and therefore never a tie either side could round
differently. Both sides then saturate to the same [-128, 127]. This was
confirmed against a real cim-opt/cim-run round trip by hand before any of
onnx_import.py's chain-walking code was written (see its own module
docstring for the exact numbers).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python"))

pytest.importorskip("onnx", reason="the ONNX front end's oracle")

from cim_frontend.onnx_import import import_model  # noqa: E402
from cim_frontend.refusal import Refusal  # noqa: E402
from onnx_fixtures import matmul_chain_model, onnx_reference_eval  # noqa: E402
from test_numerical_differential import compile_and_run, PipelineError  # noqa: E402

SEED = int(os.environ.get("CIM_TEST_SEED", "0x5eed1234abcd"), 0)


def _chain_weights(shapes, seed):
    """[K_i, N_i] int8 arrays in ONNX's own (untransposed) layout."""
    rng = np.random.default_rng(seed)
    return [rng.integers(-127, 128, size=shape, dtype=np.int64).astype(np.int8)
           for shape in shapes]


# mlp-3layer's own shape, at a size tiny-4x4's 2 tiles can actually spill
# on: three layers, none of them fitting in one tile alone, so placement
# has real eviction decisions to make across the chain.
THREE_LAYER_SHAPES = [(4, 8), (8, 6), (6, 4)]  # K0=4, N0=8=K1, N1=6=K2, N2=4


def test_a_three_layer_chain_compiles_and_matches_the_reference(cim_opt,
                                                                 cim_run):
    weights = _chain_weights(THREE_LAYER_SHAPES, SEED)
    model = matmul_chain_model(weights)
    act = np.random.default_rng(SEED + 1).integers(
        -127, 128, size=THREE_LAYER_SHAPES[0][0], dtype=np.int64).astype(np.int8)

    want = onnx_reference_eval(model, act)
    assert np.abs(want).max() < 2 ** 24, (
        "test magnitudes must stay exact through Cast(int32->float32); see "
        "onnx_import.py's module docstring on the bridge")

    text = import_model(model, act)
    try:
        outputs, _ = compile_and_run(cim_opt, cim_run, None, None, source=text)
    except PipelineError as err:
        pytest.fail(f"the imported chain failed to compile or run: {err}")

    assert np.array_equal(np.asarray(outputs), want), (
        f"compiled chain output {list(outputs)} != ONNX reference "
        f"{want.tolist()}")


def test_a_batched_three_layer_chain_compiles_and_matches_the_reference(
        cim_opt, cim_run):
    # M > 1 threads through every layer unchanged: cim.requantize does not
    # change shape (its own verifier enforces exactly that), so a batched
    # layer-0 activation makes every later layer's matmul and requantize
    # genuinely [M, ...] too, with no chain-specific code needed to make
    # this work -- see emit_chain_module's own note. Each row gets
    # independently-sampled values so a bug that mixed up rows across the
    # chain's three matmuls would produce a wrong, checkable number.
    m = 3
    weights = _chain_weights(THREE_LAYER_SHAPES, SEED + 3)
    model = matmul_chain_model(weights, act_shape=[m, THREE_LAYER_SHAPES[0][0]])
    act = np.random.default_rng(SEED + 4).integers(
        -127, 128, size=(m, THREE_LAYER_SHAPES[0][0]),
        dtype=np.int64).astype(np.int8)

    want = onnx_reference_eval(model, act)
    assert np.abs(want).max() < 2 ** 24, (
        "test magnitudes must stay exact through Cast(int32->float32); see "
        "onnx_import.py's module docstring on the bridge")

    text = import_model(model, act)
    try:
        outputs, _ = compile_and_run(cim_opt, cim_run, None, None, source=text)
    except PipelineError as err:
        pytest.fail(f"the imported batched chain failed to compile or run: {err}")

    assert np.array_equal(np.asarray(outputs), want), (
        f"compiled batched chain output {list(outputs)} != ONNX reference "
        f"{want.tolist()}")


def test_a_two_layer_chain_where_an_accumulator_actually_saturates(cim_opt,
                                                                   cim_run):
    # The bridge's clamp only matters if something actually reaches it.
    # Large-magnitude, same-signed values push layer 0's accumulator past
    # +-127, so this specifically exercises cim.requantize's clamp (and
    # QuantizeLinear's, on the oracle side) rather than only its identity
    # behaviour on small values.
    k0, n0, n1 = 8, 6, 5
    w0 = np.full((k0, n0), 100, dtype=np.int8)  # deliberately large
    w1 = np.random.default_rng(SEED + 2).integers(
        -127, 128, size=(n0, n1), dtype=np.int64).astype(np.int8)
    act = np.full(k0, 100, dtype=np.int8)  # 100*100*8 = 80000, clamps hard

    model = matmul_chain_model([w0, w1])
    want = onnx_reference_eval(model, act)

    text = import_model(model, act)
    outputs, _ = compile_and_run(cim_opt, cim_run, None, None, source=text)
    assert np.array_equal(np.asarray(outputs), want)


def test_placement_does_not_change_values_on_a_chain(cim_opt, cim_run):
    # The project's non-negotiable invariant, on the shape that actually
    # gives placement something to do across layers rather than only
    # within one.
    weights = _chain_weights(THREE_LAYER_SHAPES, SEED + 3)
    model = matmul_chain_model(weights)
    act = np.random.default_rng(SEED + 4).integers(
        -127, 128, size=THREE_LAYER_SHAPES[0][0], dtype=np.int64).astype(np.int8)
    text = import_model(model, act)

    plain, plain_profile = compile_and_run(cim_opt, cim_run, None, None,
                                           source=text, placement=False)
    placed, placed_profile = compile_and_run(cim_opt, cim_run, None, None,
                                             source=text, placement=True)

    assert np.array_equal(np.asarray(plain), np.asarray(placed))
    if "programs" in plain_profile and "programs" in placed_profile:
        assert placed_profile["programs"] <= plain_profile["programs"]


def test_a_wrong_weight_in_a_later_layer_would_actually_be_caught(cim_opt,
                                                                   cim_run):
    # Anti-vacuity for the chain path specifically: a perturbation in the
    # LAST layer (furthest from the activation) must still change the
    # compiled result, proving the whole chain is actually wired together
    # rather than the test only exercising layer 0.
    weights = _chain_weights(THREE_LAYER_SHAPES, SEED + 5)
    act = np.random.default_rng(SEED + 6).integers(
        -127, 128, size=THREE_LAYER_SHAPES[0][0], dtype=np.int64).astype(np.int8)

    base_model = matmul_chain_model(weights)
    base = onnx_reference_eval(base_model, act)

    bumped = [w.copy() for w in weights]
    bumped[-1][0, 0] = np.int8(int(bumped[-1][0, 0]) ^ 0x5A)
    bumped_model = matmul_chain_model(bumped)
    want = onnx_reference_eval(bumped_model, act)
    assert not np.array_equal(base, want), "the perturbation changed nothing"

    outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                 source=import_model(bumped_model, act))
    assert np.array_equal(np.asarray(outputs), want)


# --- chain-specific refusals -----------------------------------------------

def test_refuses_a_quantizelinear_with_nonunit_scale():
    weights = _chain_weights(THREE_LAYER_SHAPES, SEED)
    model = matmul_chain_model(weights, check=False)
    for init in model.graph.initializer:
        if init.name == "scale0":
            init.CopyFrom(
                __import__("onnx").numpy_helper.from_array(
                    np.array(2.0, dtype=np.float32), name="scale0"))
    act = np.ones(THREE_LAYER_SHAPES[0][0], dtype=np.int8)
    with pytest.raises(Refusal, match="not 1.0"):
        import_model(model, act)


def test_refuses_a_quantizelinear_with_nonzero_zero_point():
    weights = _chain_weights(THREE_LAYER_SHAPES, SEED)
    model = matmul_chain_model(weights, check=False)
    for init in model.graph.initializer:
        if init.name == "zp0":
            init.CopyFrom(
                __import__("onnx").numpy_helper.from_array(
                    np.array(5, dtype=np.int8), name="zp0"))
    act = np.ones(THREE_LAYER_SHAPES[0][0], dtype=np.int8)
    with pytest.raises(Refusal, match="non-zero zero_point"):
        import_model(model, act)


def test_refuses_a_fan_out_reading_the_bridged_activation_twice():
    # A DAG, not a chain: layer 1's activation ('q0', the bridge's own
    # output) is ALSO read by a stray extra node. This needs
    # buffer-liveness reasoning the importer does not do, so it must
    # refuse rather than silently pick one reader. Inserted BEFORE the
    # real layer 1 (rather than appended) so the graph's declared output
    # still names the real chain's last node -- otherwise the earlier
    # "graph output is not the last MatMulInteger's own output" check
    # would fire first and this test would prove nothing about fan-out.
    from onnx import helper as onnx_helper

    weights = _chain_weights(THREE_LAYER_SHAPES[:2], SEED)
    model = matmul_chain_model(weights, check=False)
    extra = onnx_helper.make_node("MatMulInteger", ["q0", "W0"], ["stray"],
                                  name="stray_reader")
    mm1_index = next(i for i, n in enumerate(model.graph.node)
                     if n.name == "mm1")
    model.graph.node.insert(mm1_index, extra)
    act = np.ones(THREE_LAYER_SHAPES[0][0], dtype=np.int8)
    with pytest.raises(Refusal, match="read by 2 node"):
        import_model(model, act)
