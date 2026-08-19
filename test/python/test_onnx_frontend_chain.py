"""Chained layers: the multi-layer counterpart to test_onnx_frontend.py.

A single MatMulInteger closes docs/roadmap.md's M2 box, but it is also the
one shape where cim-placement has nothing to reuse ACROSS: one weight
matrix, used once, is exactly the case
placement_never_changes_the_numbers_on_any_single_matmul_case already
covers. A chain of layers -- multiple weight matrices competing for a
fixed tile budget -- is what the mlp-3layer benchmark shape actually is,
and it is the first time that shape is reachable from a real model file
rather than a generated one.

THE BRIDGE, SCALE=1.0, AND A REAL CALIBRATED SCALE
--------------------------------------------------------------
Layer i's int32 accumulator has to become layer i+1's int8 activation.
cim_frontend accepts exactly one shape for that:
Cast(to=float32) -> QuantizeLinear(scale, zero_point=0, int8 out), which
it lowers to a real cim.requantize(scale, zero_point=0, effective_bits=8)
sitting between the two matmuls in the emitted MLIR. The scale may be ANY
positive constant, not just 1.0 -- a real, calibrated per-layer scale,
the standard shape of a real post-training-quantized multi-layer INT8
model.

scale=1.0 is not required by the machinery (cim.requantize's own
arithmetic is scale/zero_point/effective_bits-generic -- see
test/mlir/legalize_precision_e2e_test.cpp), it is what keeps the compiled
result exactly matching an UNQUANTIZED ONNX oracle: QuantizeLinear rounds
half-to-even and cim.requantize rounds half-away-from-zero, and those two
modes diverge only at a tie -- a value ending in exactly .5. With
scale=1.0, both sides compute round(v / 1.0) on a v that is already an
integer (an accumulator), so there is never a fractional part and
therefore never a tie either side could round differently. This was
confirmed against a real cim-opt/cim-run round trip by hand before any of
onnx_import.py's chain-walking code was written (see its own module
docstring for the exact numbers).

A REAL scale can land exactly on a tie, and at that exact point the two
rounding modes genuinely, correctly disagree by 1 -- a real
hardware-fidelity fact about cim.requantize's modeled rounding mode (a
digital requantizer or ADC readout path), not a bug. So a real-scale
chain below is checked two different ways: an ODD integer scale can PROVE
no tie ever occurs (v = scale*(n + 0.5) has no integer solution v when
scale is odd, since that requires scale/2 to be an integer), so that case
is checked for exact agreement against onnx.reference's own full
(quantized) evaluation; and a second, deliberately constructed EVEN-scale
case names the exact-tie divergence explicitly, rather than letting it
surface as a mysterious near-miss somewhere else.
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


# --- Relu directly on a bridge -----------------------------------------------

def test_a_relu_in_a_chain_matches_the_quantized_reference(cim_opt, cim_run):
    # w0 is signed and large enough that layer 0's raw accumulator has
    # both positive and negative entries for this activation, so Relu
    # actually clips something rather than passing everything through
    # unchanged -- see the fixture-assumption assertion below.
    k0, n0, n1 = 4, 6, 5
    rng = np.random.default_rng(SEED + 20)
    w0 = rng.integers(-10, 11, size=(k0, n0), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-10, 11, size=(n0, n1), dtype=np.int64).astype(np.int8)
    act = rng.integers(-10, 11, size=k0, dtype=np.int64).astype(np.int8)

    model = matmul_chain_model([w0, w1], relu_flags=[True])
    bridge_acc = act.astype(np.int64) @ w0.astype(np.int64)
    assert (bridge_acc < 0).any() and (bridge_acc > 0).any(), (
        "fixture assumption broken: layer 0's accumulator must have both "
        "a negative and a positive entry for this test to actually "
        "exercise Relu's clip, not just pass values through")

    want = onnx_reference_eval(model, act)
    outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                 source=import_model(model, act))
    assert np.array_equal(np.asarray(outputs), want), (
        f"compiled Relu chain output {list(outputs)} != ONNX reference "
        f"{want.tolist()}")


def test_a_relu_on_every_bridge_of_a_three_layer_chain(cim_opt, cim_run):
    # Both bridges flagged, on the same three-layer shape the rest of
    # this file already exercises without Relu -- proves relu_flags
    # threads through more than one bridge, not just bridge 0.
    weights = _chain_weights(THREE_LAYER_SHAPES, SEED + 21)
    model = matmul_chain_model(weights, relu_flags=[True, True])
    act = np.random.default_rng(SEED + 22).integers(
        -127, 128, size=THREE_LAYER_SHAPES[0][0], dtype=np.int64).astype(np.int8)

    want = onnx_reference_eval(model, act)
    outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                 source=import_model(model, act))
    assert np.array_equal(np.asarray(outputs), want)


def test_a_missing_relu_would_actually_be_caught(cim_opt, cim_run):
    # Anti-vacuity: compiling the SAME weights/activation with and
    # without the Relu flag must produce different output whenever the
    # bridge accumulator actually goes negative -- proving the compiled
    # module really branches on relu_flags rather than the reduce_max
    # step being emitted-but-inert (e.g. a self-max that never clips).
    k0, n0, n1 = 4, 6, 5
    rng = np.random.default_rng(SEED + 23)
    w0 = rng.integers(-10, 11, size=(k0, n0), dtype=np.int64).astype(np.int8)
    w1 = rng.integers(-10, 11, size=(n0, n1), dtype=np.int64).astype(np.int8)
    act = rng.integers(-10, 11, size=k0, dtype=np.int64).astype(np.int8)

    with_relu = matmul_chain_model([w0, w1], relu_flags=[True])
    without_relu = matmul_chain_model([w0, w1], relu_flags=[False])
    want_with = onnx_reference_eval(with_relu, act)
    want_without = onnx_reference_eval(without_relu, act)
    assert not np.array_equal(want_with, want_without), (
        "fixture assumption broken: Relu made no difference for this "
        "weight/activation pair")

    outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                 source=import_model(with_relu, act))
    assert np.array_equal(np.asarray(outputs), want_with)


# --- a real, calibrated (non-1.0) bridge scale ------------------------------

def test_a_real_calibrated_scale_matches_the_quantized_reference(cim_opt,
                                                                  cim_run):
    # An ODD integer scale makes a rounding tie IMPOSSIBLE: v/scale = n+0.5
    # for integer v and n requires scale/2 to be an integer too, which an
    # odd scale never is. So every accumulator this test's random weights
    # produce is provably rounded identically by cim.requantize
    # (half-away-from-zero) and QuantizeLinear (half-to-even) -- an exact
    # match is not luck here, it is guaranteed by the choice of scale.
    #
    # Checked against onnx.reference's own FULL evaluation (which runs the
    # graph's real QuantizeLinear, at this real scale) rather than an
    # unquantized oracle -- unlike the scale=1.0 tests above, this model's
    # own reference output IS the quantized one.
    #
    # scale=199, not a small value like 3: caught by mutation-testing this
    # test against a build of emit.py with the scale hardcoded back to
    # 1.0 -- with a small scale, this fixture's K=4..6, full-int8-range
    # weights push every accumulator's magnitude into the thousands, so
    # EVERY tried scale saturates the requantized activation to the same
    # +-128 clamp and the test passes for the wrong reason (a masked
    # mutation, not a real check). 199 is chosen large enough that this
    # fixture's specific (seeded) accumulators requantize to their exact,
    # unclamped value -- see the explicit divergence-from-scale=1 assertion
    # below, which is the actual guard against that failure mode recurring.
    scale = 199.0
    weights = _chain_weights([(4, 6), (6, 5)], SEED + 7)
    model = matmul_chain_model(weights, scales=[scale])
    act = np.random.default_rng(SEED + 8).integers(
        -127, 128, size=4, dtype=np.int64).astype(np.int8)

    want = onnx_reference_eval(model, act)
    outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                 source=import_model(model, act))
    assert np.array_equal(np.asarray(outputs), want), (
        f"compiled output {list(outputs)} != ONNX's own quantized "
        f"reference {want.tolist()} at scale={scale} -- an odd scale "
        f"should make this an exact match, never a rounding-tie mismatch")

    # The guard the paragraph above promises: if the importer/emitter ever
    # regressed to silently hardcoding scale=1.0 (this feature's entire
    # reason for existing), the compiled output would equal a scale=1.0
    # reference instead of the real, scale=199 one requested. Prove those
    # two are actually different for this fixture, so the assertion above
    # is a genuine check on the calibrated scale and not a coincidence of
    # both scales saturating to the same clamp.
    wrong_reference = onnx_reference_eval(
        matmul_chain_model(weights, scales=[1.0]), act)
    assert not np.array_equal(want, wrong_reference), (
        "the scale=199 and scale=1.0 references are identical for this "
        "fixture, which would make the assertion above pass even if the "
        "importer silently ignored the calibrated scale -- pick a scale "
        "or weights where they diverge")


def test_a_real_scale_at_an_exact_rounding_tie_documents_the_known_divergence(
        cim_opt, cim_run):
    # Deliberately constructed, not random: a single-element K=1, N=1
    # layer 0 with weight=[[1]] and activation=[1] gives a raw accumulator
    # of exactly 1. At scale=2.0, 1 / 2.0 = 0.5 EXACTLY -- a genuine tie.
    # cim.requantize (half-away-from-zero) rounds 0.5 to 1; QuantizeLinear
    # (half-to-even) rounds 0.5 to 0, since 0 is the nearest even integer.
    # This is the ONE input shape in this test file where the two engines
    # are EXPECTED to disagree, by exactly 1 -- named and asserted
    # explicitly, rather than left to surface as an unexplained failure
    # somewhere a random seed happens to hit it.
    scale = 2.0
    w0 = np.array([[1]], dtype=np.int8)   # K0=1, N0=1
    w1 = np.array([[1]], dtype=np.int8)   # K1=1, N1=1: pass layer 1's
                                          # requantized activation through
                                          # unchanged, so the disagreement
                                          # is visible in the final output.
    model = matmul_chain_model([w0, w1], scales=[scale])
    act = np.array([1], dtype=np.int8)

    onnx_result = onnx_reference_eval(model, act)
    outputs, _ = compile_and_run(cim_opt, cim_run, None, None,
                                 source=import_model(model, act))
    compiled_result = np.asarray(outputs)

    assert onnx_result.tolist() == [0], (
        "fixture assumption broken: expected ONNX's round-half-to-even "
        "to round the 0.5 tie down to 0")
    assert compiled_result.tolist() == [1], (
        "fixture assumption broken: expected cim.requantize's "
        "round-half-away-from-zero to round the 0.5 tie up to 1")
    assert compiled_result[0] - onnx_result[0] == 1, (
        f"the known rounding-tie divergence changed shape: compiled "
        f"{compiled_result.tolist()} vs ONNX {onnx_result.tolist()} -- "
        f"expected exactly a +1 difference at this tie, not something "
        f"else")


# --- chain-specific refusals -----------------------------------------------

def test_refuses_a_non_positive_scale():
    # A non-unit scale is now accepted (see the tests above) -- only a
    # genuinely invalid one (non-positive, or not a constant at all) is
    # still refused. Covers the "prove it doesn't refuse everything"
    # requirement this file's docstring names for every refusal here.
    weights = _chain_weights(THREE_LAYER_SHAPES, SEED)
    model = matmul_chain_model(weights, check=False)
    for init in model.graph.initializer:
        if init.name == "scale0":
            init.CopyFrom(
                __import__("onnx").numpy_helper.from_array(
                    np.array(-1.0, dtype=np.float32), name="scale0"))
    act = np.ones(THREE_LAYER_SHAPES[0][0], dtype=np.int8)
    with pytest.raises(Refusal, match="positive"):
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


def test_refuses_a_relu_that_is_not_directly_on_a_bridge():
    # A Relu node exists in the graph, but does not directly produce a
    # LATER layer's own input -- it sits off to the side, reading the
    # bridge's own output without being read by anything (a stray,
    # disconnected node). _strip_optional_relu's own position check only
    # recognizes a Relu at the ONE spot that matters
    # (matmul_nodes[i].input[0]'s own producer); this one is not in that
    # position, so it must still fall into the ordinary "graph also
    # contains ..." refusal, exactly like any other unrecognized op --
    # proving the allow-list addition is a position check, not a type
    # check.
    from onnx import helper as onnx_helper

    weights = _chain_weights(THREE_LAYER_SHAPES[:2], SEED)
    model = matmul_chain_model(weights, check=False)
    stray = onnx_helper.make_node("Relu", ["q0"], ["stray_relu"],
                                  name="stray_relu_node")
    model.graph.node.append(stray)
    act = np.ones(THREE_LAYER_SHAPES[0][0], dtype=np.int8)
    with pytest.raises(Refusal, match="also contains Relu"):
        import_model(model, act)


def test_refuses_a_relu_with_more_than_one_input():
    # A malformed/adversarial Relu (real ONNX Relu is strictly unary) --
    # _strip_optional_relu's own input-count check, not something ONNX's
    # checker would even allow through in practice, but the importer must
    # not assume well-formedness it has not verified.
    weights = _chain_weights(THREE_LAYER_SHAPES[:2], SEED)
    model = matmul_chain_model(weights, relu_flags=[True], check=False)
    relu_node = next(n for n in model.graph.node if n.op_type == "Relu")
    del relu_node.input[:]
    relu_node.input.extend(["q0", "q0"])
    act = np.ones(THREE_LAYER_SHAPES[0][0], dtype=np.int8)
    with pytest.raises(Refusal, match="unary"):
        import_model(model, act)
