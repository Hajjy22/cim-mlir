"""Build ONNX models in-process for the front-end tests.

Models are constructed here rather than checked in as .onnx blobs. A
protobuf blob is opaque in review; fifteen lines of `onnx.helper` are not,
and the values under test end up visible next to the assertion that uses
them -- the same reason test_numerical_differential.py spells its FIXED
cases out inline. It also means shapes can be swept instead of sampled.

`onnx.checker.check_model` runs on everything built here, so a failing
test is never "our own fixture was malformed".
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def matmul_integer_model(weight, act_name="A", weight_name="W",
                         node_name="mm", opset=10, a_zero_point=None,
                         b_zero_point=None, act_elem=TensorProto.INT8,
                         act_shape=None, extra_nodes=(), check=True):
    """A graph of one MatMulInteger: Y = A @ weight.

    `weight` is [K, N] -- ONNX's own layout, NOT the [N, K] the cim
    pipeline wants. The importer is what transposes; handing it ONNX
    layout is the whole point of these fixtures.
    """
    weight = np.asarray(weight)
    k, n = weight.shape

    inputs = [act_name, weight_name]
    initializers = [numpy_helper.from_array(weight, name=weight_name)]

    if a_zero_point is not None:
        inputs.append("a_zp")
        initializers.append(
            numpy_helper.from_array(np.asarray(a_zero_point, dtype=np.int8),
                                    name="a_zp"))
    if b_zero_point is not None:
        if a_zero_point is None:
            inputs.append("")  # optional a_zero_point left empty
        inputs.append("b_zp")
        initializers.append(
            numpy_helper.from_array(np.asarray(b_zero_point, dtype=np.int8),
                                    name="b_zp"))

    node = helper.make_node("MatMulInteger", inputs, ["Y"], name=node_name)
    graph = helper.make_graph(
        [node, *extra_nodes], "g",
        [helper.make_tensor_value_info(act_name, act_elem,
                                       act_shape or [1, k])],
        [helper.make_tensor_value_info("Y", TensorProto.INT32, [1, n])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)])
    if check:
        onnx.checker.check_model(model)
    return model


def onnx_reference_eval(model, activation, act_name="A"):
    """Evaluate with onnx.reference -- the ONNX spec's own implementation.

    This is the primary oracle: it ships inside the `onnx` package and is
    written by the people who write the specification, which is exactly
    the "written by other people for other reasons" property
    test/python/conftest.py names as the thing that makes a differential
    test worth running.
    """
    from onnx.reference import ReferenceEvaluator

    feeds = {act_name: np.asarray(activation, dtype=np.int8).reshape(1, -1)}
    return np.asarray(ReferenceEvaluator(model).run(None, feeds)[0]).ravel()


def onnxruntime_eval(model, activation, act_name="A"):
    """Evaluate with onnxruntime -- a second, independent oracle.

    A production engine rather than the reference implementation, and its
    quantized kernels are registered per dtype combination, so int8 x int8
    support is a property of the installed build rather than of the spec.
    Callers treat an exception here as "skip", never as a failure.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(),
                                   providers=["CPUExecutionProvider"])
    feeds = {act_name: np.asarray(activation, dtype=np.int8).reshape(1, -1)}
    return np.asarray(session.run(None, feeds)[0]).ravel()
