"""The front end's emitter vs. the IR shape the pipeline is tested on.

This file needs neither `onnx` nor a built `cim-opt`: it compares two
Python string builders. That is deliberate -- it means the anchor test
below runs in every environment, including the core-only builds that have
no MLIR at all.

Why the anchor matters. test/python/test_numerical_differential.py's
`build_module()` and test/mlir/pipeline_e2e_test.cpp's `buildModule()`
emit the same module shape, and every numerical claim this project makes
about the compiled pipeline is made about that shape. The ONNX front end
emits it too. If the front end's copy drifts -- most plausibly by someone
noticing that staging the activation through memref.alloc + memref.copy
looks redundant and "simplifying" it -- the symptom is not a test failure
anywhere else: cim-detect would just quietly stop seeing a candidate
(it requires exactly one constant operand), no cim.program would be
emitted, and the model would run as plain linalg. This test is what turns
that silence into a failure.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python"))

from cim_frontend.emit import emit_module, sanitize_symbol  # noqa: E402
from test_numerical_differential import build_module  # noqa: E402


# Shapes chosen the same way test_numerical_differential.py chooses them:
# against tiny-4x4's 2 tiles of 4x4, this covers a single block, N-tiled,
# K-tiled, both-tiled, and more-blocks-than-tiles. Rectangular cases are
# here on purpose -- see test_onnx_frontend.py's transpose tests.
SHAPES = [(4, 4), (8, 4), (4, 8), (8, 8), (12, 4), (16, 8)]


@pytest.mark.parametrize("n,k", SHAPES)
def test_emitter_matches_the_pipelines_own_module_shape(n, k):
    rng = np.random.default_rng(0xC1AA)
    w = rng.integers(-128, 128, size=(n, k), dtype=np.int64).astype(np.int8)
    act = rng.integers(-128, 128, size=k, dtype=np.int64).astype(np.int8)

    assert emit_module(w, act) == build_module(w, act), (
        "the ONNX front end's emitter has drifted from build_module(); see "
        "this file's docstring for why that fails silently everywhere else")


def test_emitter_rejects_a_shape_mismatch_rather_than_emitting_bad_ir():
    # A K mismatch would produce IR that fails the MLIR verifier with a
    # message about types rather than about the model, so catch it here.
    w = np.zeros((4, 8), dtype=np.int8)
    act = np.zeros(4, dtype=np.int8)
    with pytest.raises(ValueError, match="does not match"):
        emit_module(w, act)


def test_emitter_requires_the_weight_already_transposed():
    # emit_module documents that it does NOT transpose; a 1-D or 3-D weight
    # is a caller bug worth naming rather than broadcasting through.
    act = np.zeros(4, dtype=np.int8)
    with pytest.raises(ValueError, match="2-D"):
        emit_module(np.zeros(4, dtype=np.int8), act)


def test_emitter_supports_a_batched_m_greater_than_1_activation():
    # M > 1 is additive: cim-partition now tiles a real multi-row matmul
    # itself (docs/roadmap.md's M4 entry), so the emitter no longer needs
    # to refuse anything but a genuine rank/shape mismatch. See
    # test_onnx_frontend.py's own differential for the numerical proof;
    # this only pins the emitted shape.
    w = np.zeros((4, 8), dtype=np.int8)
    act = np.zeros((3, 8), dtype=np.int8)
    text = emit_module(w, act)
    assert "memref<3x8xi8>" in text
    assert "memref<3x4xi32>" in text


def test_emitter_still_rejects_an_activation_with_the_wrong_k_when_batched():
    w = np.zeros((4, 8), dtype=np.int8)
    act = np.zeros((3, 5), dtype=np.int8)
    with pytest.raises(ValueError, match="does not match"):
        emit_module(w, act)


def test_symbol_sanitization_makes_onnx_names_legal():
    taken = set()
    assert sanitize_symbol("layer1/MatMul:0", taken) == "layer1_MatMul_0"
    assert sanitize_symbol("0starts-with-digit", taken) == "_0starts_with_digit"


def test_colliding_onnx_names_get_distinct_symbols():
    # THE point of the collision set: 'w/1' and 'w_1' both sanitize to
    # 'w_1'. Sharing one memref.global between two different weight
    # matrices produces IR that verifies and numbers that are wrong.
    taken = set()
    first = sanitize_symbol("w/1", taken)
    second = sanitize_symbol("w_1", taken)
    assert first == "w_1"
    assert second == "w_1_2"
    assert first != second
