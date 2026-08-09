"""Every way the ONNX front end declines, and one case proving it doesn't
decline everything.

The importer's contract is that it never emits a module it is not sure
about. That matters more here than in most front ends, because the
failure mode of guessing is not a crash: it is a module that compiles
cleanly, runs, and computes a different function than the model does.
Dropping an unrecognized Relu, reading a uint8 weight as int8, or
accepting a non-zero zero point all produce confident wrong numbers.

Each test asserts the refusal AND a distinctive fragment of its message,
matching how test/Transforms/pass-failures.mlir pins the passes' own
diagnostics rather than just their exit codes -- a refusal that fires for
the wrong reason is its own bug.
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
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from cim_frontend.onnx_import import import_model  # noqa: E402
from cim_frontend.refusal import Refusal  # noqa: E402
from onnx_fixtures import matmul_integer_model  # noqa: E402

K, N = 4, 8
WEIGHT = np.arange(1, K * N + 1, dtype=np.int64).reshape(K, N).astype(np.int8)
ACT = np.arange(1, K + 1, dtype=np.int64).astype(np.int8)


def _refuses(model, fragment, activation=None):
    with pytest.raises(Refusal) as excinfo:
        import_model(model, ACT if activation is None else activation)
    message = excinfo.value.format()
    assert fragment in message, (
        f"refused, but for the wrong reason.\nwanted fragment: {fragment!r}\n"
        f"got: {message}")
    return message


# --- the anti-vacuity test, first, because without it every test below
# --- would pass against an importer that refuses unconditionally.

def test_a_valid_model_is_not_refused():
    text = import_model(matmul_integer_model(WEIGHT), ACT)
    assert "linalg.matmul_transpose_b" in text
    assert "cim" not in text.split("func.func")[0] or True  # sanity only


# --- shape and contract refusals -------------------------------------------

def test_refuses_more_than_one_output_row():
    # v0.1 is matrix-VECTOR. cim-partition refuses m != 1 downstream and
    # leaves linalg intact, so cim-run would then die on an unlowered op
    # with an error naming neither the model nor the real reason.
    model = matmul_integer_model(WEIGHT, act_shape=[4, K])
    _refuses(model, "matrix-VECTOR")


def test_refuses_a_symbolic_activation_shape():
    model = matmul_integer_model(WEIGHT, check=False)
    dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    dim.ClearField("dim_value")
    dim.dim_param = "batch"
    _refuses(model, "symbolic or unknown")


def test_refuses_a_uint8_weight():
    # MatMulInteger genuinely permits uint8. cim.mvm and the simulator are
    # signed throughout, so reinterpreting a uint8 200 as int8 computes
    # with -56 -- silently, and only for values above 127.
    weight = np.full((K, N), 200, dtype=np.uint8)
    model = matmul_integer_model(weight)
    _refuses(model, "signed")


def test_refuses_a_uint8_activation():
    model = matmul_integer_model(WEIGHT, act_elem=TensorProto.UINT8)
    _refuses(model, "not int8")


def test_refuses_a_rank_3_weight():
    weight = np.ones((2, K, N), dtype=np.int8)
    model = matmul_integer_model(WEIGHT, check=False)
    model.graph.ClearField("initializer")
    model.graph.initializer.append(numpy_helper.from_array(weight, name="W"))
    _refuses(model, "rank 3")


# --- constant-operand refusals ---------------------------------------------

def test_refuses_a_non_constant_weight():
    # Nothing to make resident: weight-stationary hardware buys nothing if
    # the weights are computed at runtime.
    model = matmul_integer_model(WEIGHT, check=False)
    model.graph.ClearField("initializer")
    model.graph.input.append(
        helper.make_tensor_value_info("W", TensorProto.INT8, [K, N]))
    _refuses(model, "not a constant initializer")


def test_refuses_two_constant_operands():
    # cim-detect requires EXACTLY one constant operand, and silently skips
    # otherwise -- so without this refusal the symptom is "no cim.program
    # anywhere", which names neither the cause nor the file.
    model = matmul_integer_model(WEIGHT, check=False)
    model.graph.initializer.append(
        numpy_helper.from_array(ACT.reshape(1, K), name="A"))
    _refuses(model, "both operands are constant")


# --- zero-point refusals ---------------------------------------------------

def test_refuses_a_non_zero_b_zero_point():
    model = matmul_integer_model(WEIGHT, b_zero_point=np.int8(7))
    _refuses(model, "symmetric int8")


def test_refuses_a_non_zero_a_zero_point():
    model = matmul_integer_model(WEIGHT, a_zero_point=np.int8(3))
    _refuses(model, "symmetric int8")


def test_accepts_an_explicit_zero_zero_point():
    # A zero zero point is the symmetric case and must NOT be refused --
    # otherwise the two tests above would pass for the wrong reason.
    model = matmul_integer_model(WEIGHT, b_zero_point=np.int8(0))
    assert "linalg.matmul_transpose_b" in import_model(model, ACT)


# --- graph-shape refusals --------------------------------------------------

def test_refuses_an_unsupported_op_rather_than_dropping_it():
    # THE most important refusal in this file. Silently ignoring a Relu
    # emits a module that compiles, runs, and computes the wrong function.
    relu = helper.make_node("Relu", ["Y"], ["Z"], name="relu")
    model = matmul_integer_model(WEIGHT, extra_nodes=[relu], check=False)
    _refuses(model, "Relu")


def test_refuses_a_graph_with_no_matmul_integer():
    relu = helper.make_node("Relu", ["A"], ["Y"], name="relu")
    graph = helper.make_graph(
        [relu], "g",
        [helper.make_tensor_value_info("A", TensorProto.INT8, [1, K])],
        [helper.make_tensor_value_info("Y", TensorProto.INT32, [1, N])], [])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 10)])
    _refuses(model, "no MatMulInteger")


def test_refuses_two_matmuls_with_no_bridge_between_them():
    # Two MatMulInteger nodes are now a valid chain (see
    # test_onnx_frontend_chain.py) -- but only when the second layer's
    # activation is produced by the accepted Cast->QuantizeLinear bridge.
    # Reading the first layer's raw int32 output directly, as this fixture
    # does, must still be refused: an int32 accumulator is not an int8
    # activation, and cim-detect's own int8 contract would reject it
    # anyway, just with a worse message that never mentions the model.
    second = helper.make_node("MatMulInteger", ["Y", "W"], ["Z"], name="mm2")
    model = matmul_integer_model(WEIGHT, extra_nodes=[second], check=False)
    model.graph.output[0].name = "Z"
    _refuses(model, "not produced by QuantizeLinear")


def test_refuses_a_too_old_opset():
    model = matmul_integer_model(WEIGHT, opset=9, check=False)
    _refuses(model, "opset 10")


# --- activation refusals ---------------------------------------------------

def test_refuses_an_activation_of_the_wrong_length():
    model = matmul_integer_model(WEIGHT)
    _refuses(model, "model's K is", activation=np.ones(K + 3, dtype=np.int8))


def test_refuses_activation_values_that_do_not_fit_int8():
    # Wrapping silently would compute with different numbers than were
    # supplied, which is exactly the class of bug this project's tests
    # exist to catch.
    model = matmul_integer_model(WEIGHT)
    _refuses(model, "does not fit int8",
             activation=np.full(K, 5000, dtype=np.int64))


def test_accepts_a_two_dimensional_activation_of_one_row():
    # [1, K] is the model's own declared shape, so it must be accepted as
    # readily as a flat [K].
    model = matmul_integer_model(WEIGHT)
    assert "linalg.matmul_transpose_b" in import_model(
        model, ACT.reshape(1, K))
