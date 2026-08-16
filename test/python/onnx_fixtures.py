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
                       y_scale=1.0, x_zero_point=0, w_zero_point=0,
                       y_zero_point=0, x_dtype=TensorProto.INT8,
                       y_dtype=TensorProto.INT8, bias=None,
                       strides=(1, 1), pads=(0, 0, 0, 0),
                       auto_pad="NOTSET", node_name="conv", check=True):
    """A graph of one QLinearConv: Y = requantize(conv(X, weight) [+ B]).

    `weight` is [Cout, Cin, Kh, Kw] -- ONNX's own layout, and (unlike
    matmul_integer_model's [K, N]) the SAME layout
    cim_frontend.onnx_import.load_qlinear_conv consumes directly; see
    im2col.py's own module docstring for why a conv kernel needs no
    transpose the way a matmul's weight does. `x_shape` is the
    activation's [N, Cin, H, W] -- required up front, since QLinearConv's
    output shape (and so the graph's declared output value_info) depends
    on it before any real activation values exist.

    `w_scale` is a single float (uniform across output channels) or an
    array-like of length Cout (real per-channel quantization -- see
    load_qlinear_conv's own module section header). `x_zero_point` and
    `y_zero_point` may be any int (with a matching `x_dtype`/`y_dtype` of
    TensorProto.INT8 or TensorProto.UINT8); `w_zero_point` stays 0 by
    default since load_qlinear_conv always requires it (pass a non-zero
    value here only to build a refusal-test fixture). `bias`, if given,
    is a length-Cout int32 array, ONNX's own QLinearConv bias convention.
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

    x_np_dtype = np.int8 if x_dtype == TensorProto.INT8 else np.uint8
    y_np_dtype = np.int8 if y_dtype == TensorProto.INT8 else np.uint8
    w_scale_arr = np.asarray(w_scale, dtype=np.float32)
    if w_scale_arr.ndim == 0:
        w_scale_arr = w_scale_arr.reshape(())

    x_name = "X"
    initializers = [
        numpy_helper.from_array(weight, name="W"),
        numpy_helper.from_array(np.array(x_scale, dtype=np.float32),
                                name="x_scale"),
        numpy_helper.from_array(np.array(x_zero_point, dtype=x_np_dtype),
                                name="x_zp"),
        numpy_helper.from_array(w_scale_arr, name="w_scale"),
        numpy_helper.from_array(np.array(w_zero_point, dtype=np.int8),
                                name="w_zp"),
        numpy_helper.from_array(np.array(y_scale, dtype=np.float32),
                                name="y_scale"),
        numpy_helper.from_array(np.array(y_zero_point, dtype=y_np_dtype),
                                name="y_zp"),
    ]
    node_inputs = [x_name, "x_scale", "x_zp", "W", "w_scale", "w_zp",
                   "y_scale", "y_zp"]
    if bias is not None:
        initializers.append(numpy_helper.from_array(
            np.asarray(bias, dtype=np.int32), name="B"))
        node_inputs.append("B")
    node_kwargs = {"strides": list(strides)}
    if auto_pad == "NOTSET":
        node_kwargs["pads"] = list(pads)
    else:
        node_kwargs["auto_pad"] = auto_pad
    node = helper.make_node(
        "QLinearConv", node_inputs, ["Y"], name=node_name, **node_kwargs)
    graph = helper.make_graph(
        [node], "conv_g",
        [helper.make_tensor_value_info(x_name, x_dtype, list(x_shape))],
        [helper.make_tensor_value_info(
            "Y", y_dtype, [n, cout, out_h, out_w])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    if check:
        onnx.checker.check_model(model)
    return model


def small_multi_layer_cnn_model(seed=0, check=True):
    """Three QLinearConv layers with two MaxPool ops between them --
    conv/pool/conv/pool/conv, loosely the early-layer shape of a real
    small classification CNN (a SqueezeNet-style stem: a few 3x3 convs
    narrowing spatial size, expanding channels, capped by a 1x1
    "classifier-ish" conv). HAND-BUILT, not downloaded from the ONNX model
    zoo: this session's sandboxed network policy returns 403 for a raw
    GitHub content fetch, so `analyze.py`'s own multi-op-graph test
    coverage cannot depend on one being reachable. What matters for that
    coverage is real either way -- multiple offloadable layers of
    different [K, N] competing for tile residency, interleaved with ops
    that have no resident weights -- not that the specific numbers came
    from a trained model.

    Backs test/python/test_analyze.py's checked-in JSON workload fixture
    (test/workloads/small-cnn-workload.json) the same way
    onnx_fixtures.matmul_integer_model backs
    test/Transforms/onnx-imported-matmul.mlir: a test regenerates the
    fixture from this function and diffs it, so it cannot silently go
    stale (test_analyze.py::test_checked_in_workload_fixture_is_current).
    """
    rng = np.random.default_rng(seed)

    def conv_layer(name, x_name, cout, cin, kh, kw):
        w = rng.integers(-3, 4, size=(cout, cin, kh, kw),
                         dtype=np.int64).astype(np.int8)
        inits = [
            numpy_helper.from_array(w, name=f"{name}_w"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32),
                                    name=f"{name}_xs"),
            numpy_helper.from_array(np.array(0, dtype=np.int8),
                                    name=f"{name}_xzp"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32),
                                    name=f"{name}_ws"),
            numpy_helper.from_array(np.array(0, dtype=np.int8),
                                    name=f"{name}_wzp"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32),
                                    name=f"{name}_ys"),
            numpy_helper.from_array(np.array(0, dtype=np.int8),
                                    name=f"{name}_yzp"),
        ]
        node = helper.make_node(
            "QLinearConv",
            [x_name, f"{name}_xs", f"{name}_xzp", f"{name}_w", f"{name}_ws",
             f"{name}_wzp", f"{name}_ys", f"{name}_yzp"],
            [f"{name}_y"], name=name, strides=[1, 1], pads=[0, 0, 0, 0])
        return node, inits

    def pool_layer(name, x_name):
        return helper.make_node("MaxPool", [x_name], [f"{name}_y"],
                                name=name, kernel_shape=[2, 2],
                                strides=[2, 2])

    conv1, init1 = conv_layer("conv1", "X", cout=8, cin=3, kh=3, kw=3)
    pool1 = pool_layer("pool1", "conv1_y")
    conv2, init2 = conv_layer("conv2", "pool1_y", cout=16, cin=8, kh=3, kw=3)
    pool2 = pool_layer("pool2", "conv2_y")
    conv3, init3 = conv_layer("conv3", "pool2_y", cout=32, cin=16, kh=1,
                              kw=1)

    graph = helper.make_graph(
        [conv1, pool1, conv2, pool2, conv3], "small_cnn",
        [helper.make_tensor_value_info("X", TensorProto.INT8, [1, 3, 10, 10])],
        [helper.make_tensor_value_info("conv3_y", TensorProto.INT8,
                                       [1, 32, 1, 1])],
        init1 + init2 + init3,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    if check:
        onnx.checker.check_model(model)
    return model


def _as_activation_feed(activation, dtype=np.int8):
    """[K] promotes to [1, K]; a real [M, K] batch passes through as-is --
    same normalization `cim_frontend.onnx_import._validate_activation`
    applies, so both oracles and the compiled pipeline see the same shape
    for the same input. `dtype` defaults to int8 (every non-conv fixture's
    activation); pass `np.uint8` for a QLinearConv model whose `X` is
    declared uint8 (a real, asymmetric-quantization convention -- see
    onnx_fixtures.qlinear_conv_model's own `x_dtype` parameter)."""
    activation = np.asarray(activation, dtype=dtype)
    return activation.reshape(1, -1) if activation.ndim == 1 else activation


def onnx_reference_eval(model, activation, act_name="A", activation_dtype=np.int8):
    """Evaluate with onnx.reference -- the ONNX spec's own implementation.

    This is the primary oracle: it ships inside the `onnx` package and is
    written by the people who write the specification, which is exactly
    the "written by other people for other reasons" property
    test/python/conftest.py names as the thing that makes a differential
    test worth running.
    """
    from onnx.reference import ReferenceEvaluator

    feeds = {act_name: _as_activation_feed(activation, activation_dtype)}
    return np.asarray(ReferenceEvaluator(model).run(None, feeds)[0]).ravel()


def onnxruntime_eval(model, activation, act_name="A", activation_dtype=np.int8):
    """Evaluate with onnxruntime -- a second, independent oracle.

    A production engine rather than the reference implementation, and its
    quantized kernels are registered per dtype combination, so int8 x int8
    support is a property of the installed build rather than of the spec.
    Callers treat an exception here as "skip", never as a failure.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(),
                                   providers=["CPUExecutionProvider"])
    feeds = {act_name: _as_activation_feed(activation, activation_dtype)}
    return np.asarray(session.run(None, feeds)[0]).ravel()
