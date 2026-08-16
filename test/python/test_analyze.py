"""cim_frontend.analyze: the permissive graph walker behind
`cim-import-onnx --emit-workload`.

Unlike test_onnx_frontend*.py, which pins what the STRICT (compiling)
loaders refuse, this file pins that the ANALYSIS walker never refuses a
whole graph -- every node is classified as offloadable or skipped, and
the walk always completes -- while still applying the same per-operand
correctness checks those loaders use, just per node instead of per graph.
See analyze.py's own module docstring for why that split is safe.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python"))

pytest.importorskip("onnx", reason="the ONNX front end's oracle")

import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from cim_frontend.analyze import analyze_model  # noqa: E402
from cim_frontend.refusal import Refusal  # noqa: E402
from onnx_fixtures import (matmul_integer_model, qlinear_conv_model,  # noqa: E402
                           small_multi_layer_cnn_model)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


# --- the anti-vacuity test, first: without it every "N skipped" assertion
# --- below would pass against a walker that skips everything.

def test_a_valid_single_matmul_is_offloaded_not_skipped():
    report = analyze_model(matmul_integer_model(np.ones((8, 4), dtype=np.int8)))
    assert len(report["layers"]) == 1
    assert report["layers"][0] == {"name": "mm", "op_type": "MatMulInteger",
                                   "k": 8, "n": 4}
    assert report["skipped"] == []


def test_a_valid_conv_is_offloaded_not_skipped():
    weight = np.ones((6, 3, 2, 2), dtype=np.int8)
    report = analyze_model(qlinear_conv_model(weight, x_shape=(1, 3, 5, 5)))
    assert len(report["layers"]) == 1
    layer = report["layers"][0]
    assert layer["op_type"] == "QLinearConv"
    assert layer["k"] == 3 * 2 * 2  # Cin * Kh * Kw
    assert layer["n"] == 6          # Cout


# --- the walk never refuses the graph -----------------------------------

def _mixed_graph(offload_node, other_node, other_input, output_name):
    weight = np.ones((8, 4), dtype=np.int8)
    graph = helper.make_graph(
        [offload_node, other_node], "mixed",
        [helper.make_tensor_value_info("A", TensorProto.INT8, [1, 8])],
        [helper.make_tensor_value_info(output_name, TensorProto.INT32, [1, 4])],
        [numpy_helper.from_array(weight, name="W")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_unrecognized_op_is_skipped_with_a_named_reason():
    mm = helper.make_node("MatMulInteger", ["A", "W"], ["Y"], name="mm1")
    relu = helper.make_node("Relu", ["Y"], ["Z"], name="relu1")
    model = _mixed_graph(mm, relu, "Y", "Z")

    report = analyze_model(model)
    assert len(report["layers"]) == 1
    assert len(report["skipped"]) == 1
    skip = report["skipped"][0]
    assert skip["name"] == "relu1"
    assert skip["op_type"] == "Relu"
    assert "Relu" in skip["reason"]
    assert "not emitted" not in skip["reason"].lower()  # see analyze._reason


def test_a_matmul_of_two_constants_is_skipped_not_crashed():
    a = np.ones((1, 8), dtype=np.int8)
    w = np.ones((8, 4), dtype=np.int8)
    mm = helper.make_node("MatMulInteger", ["A", "W"], ["Y"], name="mm1")
    graph = helper.make_graph(
        [mm], "both_const", [],
        [helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4])],
        [numpy_helper.from_array(a, name="A"), numpy_helper.from_array(w, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    report = analyze_model(model)
    assert report["layers"] == []
    assert len(report["skipped"]) == 1
    assert "both operands are constant" in report["skipped"][0]["reason"]


def test_non_int8_weight_is_skipped():
    weight = np.ones((8, 4), dtype=np.float32)
    mm = helper.make_node("MatMulInteger", ["A", "W"], ["Y"], name="mm1")
    graph = helper.make_graph(
        [mm], "g",
        [helper.make_tensor_value_info("A", TensorProto.INT8, [1, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4])],
        [numpy_helper.from_array(weight, name="W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    report = analyze_model(model)
    assert report["layers"] == []
    assert "float32" in report["skipped"][0]["reason"]


def test_nonzero_zero_point_is_skipped():
    weight = np.ones((8, 4), dtype=np.int8)
    zp = numpy_helper.from_array(np.array(3, dtype=np.int8), name="a_zp")
    mm = helper.make_node("MatMulInteger", ["A", "W", "a_zp"], ["Y"], name="mm1")
    graph = helper.make_graph(
        [mm], "g",
        [helper.make_tensor_value_info("A", TensorProto.INT8, [1, 8])],
        [helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 4])],
        [numpy_helper.from_array(weight, name="W"), zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    report = analyze_model(model)
    assert report["layers"] == []
    assert "a_zero_point" in report["skipped"][0]["reason"]


# --- grouped/dilated conv: the one place analysis is DELIBERATELY more
# --- permissive than compilation, because the shape is real either way.

def test_grouped_conv_is_offloaded_despite_not_being_compilable_today():
    # ONNX already stores a grouped conv's weight as [Cout, Cin/group, Kh,
    # Kw] -- the real per-filter footprint -- so k = Cin/group * Kh * Kw is
    # correct without needing to execute the grouped convolution at all.
    weight = np.ones((6, 3, 2, 2), dtype=np.int8)  # group=2, Cin/group=3
    x_scale = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="x_scale")
    x_zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="x_zp")
    w_scale = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="w_scale")
    w_zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="w_zp")
    y_scale = numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="y_scale")
    y_zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="y_zp")
    node = helper.make_node(
        "QLinearConv", ["X", "x_scale", "x_zp", "W", "w_scale", "w_zp",
                        "y_scale", "y_zp"], ["Y"], name="gconv", group=2,
        pads=[0, 0, 0, 0], strides=[1, 1])
    graph = helper.make_graph(
        [node], "grouped",
        [helper.make_tensor_value_info("X", TensorProto.INT8, [1, 6, 5, 5])],
        [helper.make_tensor_value_info("Y", TensorProto.INT8, [1, 6, 4, 4])],
        [numpy_helper.from_array(weight, name="W"), x_scale, x_zp, w_scale,
         w_zp, y_scale, y_zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    report = analyze_model(model)
    assert len(report["layers"]) == 1
    assert report["layers"][0]["k"] == 3 * 2 * 2
    assert report["layers"][0]["n"] == 6
    assert report["skipped"] == []


# --- graph-level (not per-node) refusals still raise -----------------------

def test_unusable_opset_refuses_the_whole_call():
    model = matmul_integer_model(np.ones((8, 4), dtype=np.int8), opset=9,
                                 check=False)
    with pytest.raises(Refusal):
        analyze_model(model)


# --- the honesty requirement: `note` always states counts in words ---------

def test_note_names_both_counts_even_when_nothing_was_skipped():
    report = analyze_model(matmul_integer_model(np.ones((8, 4), dtype=np.int8)))
    assert "1 offloadable layer" in report["note"]
    assert "0 other op" in report["note"]
    assert "NOT end-to-end inference cost" in report["note"]


# --- the multi-layer fixture also backs the checked-in JSON below ----------

def test_multi_layer_cnn_fixture_has_three_layers_two_skips():
    report = analyze_model(small_multi_layer_cnn_model())
    assert [l["op_type"] for l in report["layers"]] == ["QLinearConv"] * 3
    assert [s["op_type"] for s in report["skipped"]] == ["MaxPool", "MaxPool"]


def test_checked_in_workload_fixture_is_current():
    """test/workloads/small-cnn-workload.json must still be what
    analyze_model produces for small_multi_layer_cnn_model() -- same
    convention as test_onnx_frontend.py's
    test_checked_in_lit_fixture_is_current for the MLIR fixture: a
    checked-in artifact that nothing regenerates silently rots, so this
    regenerates it and compares."""
    report = analyze_model(small_multi_layer_cnn_model(), model_path="small-cnn.onnx")
    fixture = os.path.join(REPO_ROOT, "test", "workloads",
                           "small-cnn-workload.json")
    with open(fixture) as handle:
        checked_in = json.load(handle)
    assert report == checked_in, (
        "test/workloads/small-cnn-workload.json is stale; regenerate it "
        "from onnx_fixtures.small_multi_layer_cnn_model() via "
        "cim_frontend.analyze.analyze_model()")


# --- the CLI wiring itself, in-process being insufficient: an argparse ----
# --- regression (e.g. --emit-workload silently requiring --input again) --
# --- would only show up by actually invoking __main__.

def test_cli_emit_workload_needs_no_activation(tmp_path):
    model = matmul_integer_model(np.ones((8, 4), dtype=np.int8))
    model_path = tmp_path / "mm.onnx"
    onnx.save(model, str(model_path))
    out_path = tmp_path / "w.json"

    result = subprocess.run(
        [sys.executable, "-m", "cim_frontend", str(model_path),
         "--emit-workload", "-o", str(out_path)],
        cwd=os.path.join(REPO_ROOT, "python"),
        env={**os.environ, "PYTHONPATH": os.path.join(REPO_ROOT, "python")},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr

    with open(out_path) as handle:
        report = json.load(handle)
    assert report["layers"][0]["k"] == 8
    assert report["layers"][0]["n"] == 4


def test_cli_without_emit_workload_still_requires_an_activation(tmp_path):
    model = matmul_integer_model(np.ones((8, 4), dtype=np.int8))
    model_path = tmp_path / "mm.onnx"
    onnx.save(model, str(model_path))

    result = subprocess.run(
        [sys.executable, "-m", "cim_frontend", str(model_path)],
        cwd=os.path.join(REPO_ROOT, "python"),
        env={**os.environ, "PYTHONPATH": os.path.join(REPO_ROOT, "python")},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert "no activation supplied" in result.stderr
