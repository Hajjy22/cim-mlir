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


def matmul_chain_model(weights, act_name="A", check=True, act_shape=None,
                       scales=None):
    """A linear chain of MatMulInteger nodes, each bridged into the next by
    the one pattern python/cim_frontend/onnx_import.py accepts:
    Cast(to=float32) -> QuantizeLinear(scale, zero_point=0, int8 out).

    `weights` is a list of [K_i, N_i] int8 arrays in ONNX's own layout
    (NOT transposed), with K_i == N_(i-1) for i > 0 -- same convention as
    matmul_integer_model, so both fixtures hand the importer the same kind
    of un-transposed input. `act_shape`, like matmul_integer_model's own,
    overrides the declared [1, K_0] activation shape -- e.g. [M, K_0] for
    a batched (M > 1) chain. `scales`, if given, is a list of
    `len(weights) - 1` positive floats, one per bridge -- a real,
    calibrated scale instead of the default 1.0 (see onnx_import.py's own
    "WHY THIS EXACT BRIDGE" note on what a non-1.0 scale changes).
    """
    weights = [np.asarray(w) for w in weights]
    if scales is None:
        scales = [1.0] * (len(weights) - 1)
    if len(scales) != len(weights) - 1:
        raise ValueError(
            f"scales must have one entry per bridge (len(weights) - 1 = "
            f"{len(weights) - 1}), got {len(scales)}")
    nodes = []
    initializers = []
    cur = act_name

    for i, w in enumerate(weights):
        w_name = f"W{i}"
        y_name = f"Y{i}"
        initializers.append(numpy_helper.from_array(w, name=w_name))
        nodes.append(helper.make_node("MatMulInteger", [cur, w_name],
                                      [y_name], name=f"mm{i}"))

        if i < len(weights) - 1:
            float_name = f"f{i}"
            q_name = f"q{i}"
            scale_name = f"scale{i}"
            zp_name = f"zp{i}"
            initializers.append(numpy_helper.from_array(
                np.array(scales[i], dtype=np.float32), name=scale_name))
            initializers.append(numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=zp_name))
            nodes.append(helper.make_node(
                "Cast", [y_name], [float_name], to=TensorProto.FLOAT,
                name=f"cast{i}"))
            nodes.append(helper.make_node(
                "QuantizeLinear", [float_name, scale_name, zp_name],
                [q_name], name=f"quant{i}"))
            cur = q_name

    k0 = weights[0].shape[0]
    n_last = weights[-1].shape[1]
    m = (act_shape or [1, k0])[0]
    graph = helper.make_graph(
        nodes, "chain",
        [helper.make_tensor_value_info(act_name, TensorProto.INT8,
                                       act_shape or [1, k0])],
        [helper.make_tensor_value_info(f"Y{len(weights) - 1}",
                                       TensorProto.INT32, [m, n_last])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    if check:
        onnx.checker.check_model(model)
    return model


def qlinear_conv_model(weight, x_shape, x_scale=1.0, w_scale=1.0,
                       y_scale=1.0, strides=(1, 1), pads=(0, 0, 0, 0),
                       auto_pad="NOTSET", node_name="conv", check=True):
    """A graph of one QLinearConv: Y = requantize(conv(X, weight)).

    `weight` is [Cout, Cin, Kh, Kw] -- ONNX's own layout, and (unlike
    matmul_integer_model's [K, N]) the SAME layout
    cim_frontend.onnx_import.load_qlinear_conv consumes directly; see
    im2col.py's own module docstring for why a conv kernel needs no
    transpose the way a matmul's weight does. `x_shape` is the
    activation's [N, Cin, H, W] -- required up front, since QLinearConv's
    output shape (and so the graph's declared output value_info) depends
    on it before any real activation values exist.

    Zero points are fixed at 0 throughout (symmetric quantization, the
    only case `load_qlinear_conv` accepts) and there is no bias operand --
    both match that function's own documented scope.
    """
    weight = np.asarray(weight)
    cout, cin, kh, kw = weight.shape
    n, x_cin, h, w = x_shape
    if x_cin != cin:
        raise ValueError(
            f"x_shape's Cin ({x_cin}) does not match weight's Cin ({cin})")

    stride_h, stride_w = strides
    if auto_pad == "NOTSET":
        pad_top, pad_left, pad_bottom, pad_right = pads
    else:
        pad_top = pad_left = pad_bottom = pad_right = 0
    out_h = (h + pad_top + pad_bottom - kh) // stride_h + 1
    out_w = (w + pad_left + pad_right - kw) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"non-positive output size ({out_h}, {out_w}) for the given "
            f"x_shape/weight/strides/pads")

    x_name = "X"
    initializers = [
        numpy_helper.from_array(weight, name="W"),
        numpy_helper.from_array(np.array(x_scale, dtype=np.float32),
                                name="x_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="x_zp"),
        numpy_helper.from_array(np.array(w_scale, dtype=np.float32),
                                name="w_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="w_zp"),
        numpy_helper.from_array(np.array(y_scale, dtype=np.float32),
                                name="y_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="y_zp"),
    ]
    node_kwargs = {"strides": list(strides)}
    if auto_pad == "NOTSET":
        node_kwargs["pads"] = list(pads)
    else:
        node_kwargs["auto_pad"] = auto_pad
    node = helper.make_node(
        "QLinearConv",
        [x_name, "x_scale", "x_zp", "W", "w_scale", "w_zp", "y_scale",
         "y_zp"],
        ["Y"], name=node_name, **node_kwargs)
    graph = helper.make_graph(
        [node], "conv_g",
        [helper.make_tensor_value_info(x_name, TensorProto.INT8,
                                       list(x_shape))],
        [helper.make_tensor_value_info(
            "Y", TensorProto.INT8, [n, cout, out_h, out_w])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    if check:
        onnx.checker.check_model(model)
    return model


def _as_activation_feed(activation):
    """[K] promotes to [1, K]; a real [M, K] batch passes through as-is --
    same normalization `cim_frontend.onnx_import._validate_activation`
    applies, so both oracles and the compiled pipeline see the same shape
    for the same input."""
    activation = np.asarray(activation, dtype=np.int8)
    return activation.reshape(1, -1) if activation.ndim == 1 else activation


def onnx_reference_eval(model, activation, act_name="A"):
    """Evaluate with onnx.reference -- the ONNX spec's own implementation.

    This is the primary oracle: it ships inside the `onnx` package and is
    written by the people who write the specification, which is exactly
    the "written by other people for other reasons" property
    test/python/conftest.py names as the thing that makes a differential
    test worth running.
    """
    from onnx.reference import ReferenceEvaluator

    feeds = {act_name: _as_activation_feed(activation)}
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
    feeds = {act_name: _as_activation_feed(activation)}
    return np.asarray(session.run(None, feeds)[0]).ravel()
