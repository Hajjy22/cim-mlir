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
                       dilations=(1, 1),
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
    `dilations` defaults to (1, 1) -- ordinary convolution -- matching
    every call site before dilation support existed.
    """
    weight = np.asarray(weight)
    cout, cin, kh, kw = weight.shape
    n, x_cin, h, w = x_shape
    if x_cin != cin:
        raise ValueError(
            f"x_shape's Cin ({x_cin}) does not match weight's Cin ({cin})")

    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations
    if auto_pad == "NOTSET":
        pad_top, pad_left, pad_bottom, pad_right = pads
    else:
        pad_top = pad_left = pad_bottom = pad_right = 0
    dilated_kh = (kh - 1) * dilation_h + 1
    dilated_kw = (kw - 1) * dilation_w + 1
    out_h = (h + pad_top + pad_bottom - dilated_kh) // stride_h + 1
    out_w = (w + pad_left + pad_right - dilated_kw) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"non-positive output size ({out_h}, {out_w}) for the given "
            f"x_shape/weight/strides/pads/dilations")

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
    node_kwargs = {"strides": list(strides), "dilations": list(dilations)}
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


def conv_matmul_chain_model(conv_weight, x_shape, matmul_weights,
                            x_scale=1.0, w_scale=1.0, y_scale=1.0,
                            x_zero_point=0, x_dtype=TensorProto.INT8,
                            strides=(1, 1), pads=(0, 0, 0, 0),
                            dilations=(1, 1), bridge_scales=None,
                            check=True):
    """A QLinearConv (layer 0) feeding one or more MatMulInteger layers,
    bridged the way cim_frontend.onnx_import.load_conv_matmul_chain reads:
    Transpose(perm=[0, 2, 3, 1]) -> Reshape([M, Cout]) between the conv and
    the first matmul -- REQUIRED, not just tolerated, because ONNX's own
    MatMulInteger reference kernel does N-D broadcast matmul rather than
    requiring 2-D operands, so wiring the conv's raw [N, Cout, OutH, OutW]
    output directly into a MatMulInteger would silently contract the wrong
    axes (confirmed by hand -- see load_conv_matmul_chain's own module
    section header) rather than erroring. Contrast with the
    Cast/QuantizeLinear pair matmul_chain_model emits BETWEEN matmul
    layers, still used here for every bridge after the first (QLinearConv
    already quantizes its own output in-node, so no Cast/QuantizeLinear
    bridge exists between the conv and the first matmul).

    `conv_weight` is [Cout, Cin, Kh, Kw] (ONNX layout, as qlinear_conv_model
    takes). `matmul_weights` is a list of [K_i, N_i] arrays (ONNX layout,
    as matmul_chain_model takes), with K_0 == conv_weight's own Cout.
    `bridge_scales`, if given, is a list of `len(matmul_weights) - 1`
    floats for the bridges BETWEEN matmul layers (never between the conv
    and the first matmul -- that bridge's scale is always derived from
    `y_scale / (x_scale * w_scale)`, exactly as a standalone QLinearConv's
    own `trailing_requantize` is).

    Unlike qlinear_conv_model's own general y_zero_point/w_scale/bias
    support, this fixture always uses y_zero_point=0, a scalar w_scale,
    and no bias on the conv layer -- load_conv_matmul_chain requires
    exactly that for a convolution chained into further layers (see its
    own module section header).
    """
    conv_weight = np.asarray(conv_weight)
    cout, cin, kh, kw = conv_weight.shape
    n, x_cin, h, w = x_shape
    if x_cin != cin:
        raise ValueError(
            f"x_shape's Cin ({x_cin}) does not match conv_weight's Cin "
            f"({cin})")
    matmul_weights = [np.asarray(w) for w in matmul_weights]
    if matmul_weights[0].shape[0] != cout:
        raise ValueError(
            f"matmul_weights[0]'s K ({matmul_weights[0].shape[0]}) does "
            f"not match the conv's own Cout ({cout})")
    if bridge_scales is None:
        bridge_scales = [1.0] * (len(matmul_weights) - 1)
    if len(bridge_scales) != len(matmul_weights) - 1:
        raise ValueError(
            f"bridge_scales must have one entry per inter-matmul bridge "
            f"(len(matmul_weights) - 1 = {len(matmul_weights) - 1}), got "
            f"{len(bridge_scales)}")

    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations
    pad_top, pad_left, pad_bottom, pad_right = pads
    dilated_kh = (kh - 1) * dilation_h + 1
    dilated_kw = (kw - 1) * dilation_w + 1
    out_h = (h + pad_top + pad_bottom - dilated_kh) // stride_h + 1
    out_w = (w + pad_left + pad_right - dilated_kw) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"non-positive output size ({out_h}, {out_w}) for the given "
            f"x_shape/conv_weight/strides/pads/dilations")

    x_name = "X"
    x_np_dtype = np.int8 if x_dtype == TensorProto.INT8 else np.uint8
    initializers = [
        numpy_helper.from_array(conv_weight, name="conv_W"),
        numpy_helper.from_array(np.array(x_scale, dtype=np.float32),
                                name="conv_xs"),
        numpy_helper.from_array(np.array(x_zero_point, dtype=x_np_dtype),
                                name="conv_xzp"),
        numpy_helper.from_array(np.array(w_scale, dtype=np.float32),
                                name="conv_ws"),
        numpy_helper.from_array(np.array(0, dtype=np.int8),
                                name="conv_wzp"),
        numpy_helper.from_array(np.array(y_scale, dtype=np.float32),
                                name="conv_ys"),
        numpy_helper.from_array(np.array(0, dtype=np.int8),
                                name="conv_yzp"),
    ]
    conv_node = helper.make_node(
        "QLinearConv",
        [x_name, "conv_xs", "conv_xzp", "conv_W", "conv_ws", "conv_wzp",
         "conv_ys", "conv_yzp"],
        ["conv_y"], name="conv0", strides=list(strides),
        pads=list(pads), dilations=list(dilations))
    transpose_node = helper.make_node(
        "Transpose", ["conv_y"], ["conv_y_nhwc"], perm=[0, 2, 3, 1],
        name="transpose0")
    m = n * out_h * out_w
    initializers.append(numpy_helper.from_array(
        np.array([m, cout], dtype=np.int64), name="reshape0_shape"))
    reshape_node = helper.make_node(
        "Reshape", ["conv_y_nhwc", "reshape0_shape"], ["conv_y_flat"],
        name="reshape0")

    nodes = [conv_node, transpose_node, reshape_node]
    cur = "conv_y_flat"
    for i, mw in enumerate(matmul_weights):
        w_name = f"mmW{i}"
        y_name = f"mmY{i}"
        initializers.append(numpy_helper.from_array(mw, name=w_name))
        nodes.append(helper.make_node("MatMulInteger", [cur, w_name],
                                      [y_name], name=f"mm{i}"))
        if i < len(matmul_weights) - 1:
            float_name, q_name = f"mmf{i}", f"mmq{i}"
            scale_name, zp_name = f"mmscale{i}", f"mmzp{i}"
            initializers.append(numpy_helper.from_array(
                np.array(bridge_scales[i], dtype=np.float32),
                name=scale_name))
            initializers.append(numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=zp_name))
            nodes.append(helper.make_node(
                "Cast", [y_name], [float_name], to=TensorProto.FLOAT,
                name=f"mmcast{i}"))
            nodes.append(helper.make_node(
                "QuantizeLinear", [float_name, scale_name, zp_name],
                [q_name], name=f"mmquant{i}"))
            cur = q_name

    n_last = matmul_weights[-1].shape[1]
    graph = helper.make_graph(
        nodes, "conv_matmul_chain",
        [helper.make_tensor_value_info(x_name, x_dtype, list(x_shape))],
        [helper.make_tensor_value_info(
            f"mmY{len(matmul_weights) - 1}", TensorProto.INT32,
            [m, n_last])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    if check:
        onnx.checker.check_model(model)
    return model


def conv_chain_model(weights, x_shape, x_scale=1.0, x_zero_point=0,
                     x_dtype=TensorProto.INT8, w_scales=None,
                     y_scales=None, strides=None, pads=None,
                     dilations=None, last_bias=None,
                     last_y_zero_point=0, last_y_dtype=TensorProto.INT8,
                     check=True):
    """A chain of two or more QLinearConv nodes, connected DIRECTLY --
    conv_i's own "Y" feeds conv_(i+1)'s own "X" with no bridge node
    between them, the way cim_frontend.onnx_import.load_conv_chain reads
    (see its own module section header for why no bridge is needed here,
    unlike conv_matmul_chain_model's required Transpose/Reshape, or
    matmul_chain_model's required Cast/QuantizeLinear).

    `weights[i]` is layer i's own [Cout_i, Cin_i, Kh_i, Kw_i] weight (ONNX
    layout -- qlinear_conv_model's own convention), with Cin_i ==
    Cout_(i-1) for i > 0. `w_scales[i]`/`y_scales[i]`/`strides[i]`/
    `pads[i]`/`dilations[i]` are layer i's own; `w_scales`/`y_scales`
    default to 1.0 for every layer, `strides`/`dilations` to (1, 1),
    `pads` to (0, 0, 0, 0), if not given.

    Only the LAST layer may have a non-zero y_zero_point
    (`last_y_zero_point`, with matching `last_y_dtype`), a per-channel
    w_scale (an array-like `w_scales[-1]` of length Cout of the last
    layer), or a bias (`last_bias`, length-Cout int32) -- every earlier
    layer is forced to y_zero_point=0, a scalar w_scale, and no bias,
    matching load_conv_chain's own restriction (an intermediate bridge is
    ONE unconditional scalar cim.requantize, with no cim.reduce_partial
    step available). Every layer but the FIRST also gets x_zero_point=0,
    x_dtype=INT8 -- not a parameter here, since load_conv_chain requires
    it (an intermediate activation arrives already zero-centered, with no
    zero-point shift available in the emitted IR) and a fixture that could
    build a graph load_conv_chain would refuse defeats the point of a
    "matches the reference" fixture. Only layer 0's `x_zero_point`/
    `x_dtype` are configurable, exactly like qlinear_conv_model's own.
    """
    weights = [np.asarray(w) for w in weights]
    n_layers = len(weights)
    if n_layers < 2:
        raise ValueError("conv_chain_model needs at least two conv layers")
    if w_scales is None:
        w_scales = [1.0] * n_layers
    if y_scales is None:
        y_scales = [1.0] * n_layers
    if strides is None:
        strides = [(1, 1)] * n_layers
    if pads is None:
        pads = [(0, 0, 0, 0)] * n_layers
    if dilations is None:
        dilations = [(1, 1)] * n_layers
    for name, seq in (("w_scales", w_scales), ("y_scales", y_scales),
                      ("strides", strides), ("pads", pads),
                      ("dilations", dilations)):
        if len(seq) != n_layers:
            raise ValueError(f"{name} must have one entry per layer "
                             f"({n_layers}), got {len(seq)}")

    n, cin0, h0, w0 = x_shape
    if weights[0].shape[1] != cin0:
        raise ValueError(
            f"x_shape's Cin ({cin0}) does not match layer 0's own Cin "
            f"({weights[0].shape[1]})")
    for i in range(1, n_layers):
        if weights[i].shape[1] != weights[i - 1].shape[0]:
            raise ValueError(
                f"layer {i}'s Cin ({weights[i].shape[1]}) does not match "
                f"layer {i - 1}'s Cout ({weights[i - 1].shape[0]})")

    x_name = "X"
    x_np_dtype = np.int8 if x_dtype == TensorProto.INT8 else np.uint8
    initializers = []
    nodes = []
    cur_name = x_name
    cur_h, cur_w = h0, w0
    first_x_dtype = x_dtype
    layer_y_dtype = TensorProto.INT8

    for i, w in enumerate(weights):
        cout, cin, kh, kw = w.shape
        stride_h, stride_w = strides[i]
        pad_top, pad_left, pad_bottom, pad_right = pads[i]
        dilation_h, dilation_w = dilations[i]
        dilated_kh = (kh - 1) * dilation_h + 1
        dilated_kw = (kw - 1) * dilation_w + 1
        out_h = (cur_h + pad_top + pad_bottom - dilated_kh) // stride_h + 1
        out_w = (cur_w + pad_left + pad_right - dilated_kw) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"layer {i}: non-positive output size ({out_h}, {out_w}) "
                f"for the given x_shape/weights/strides/pads/dilations")

        is_first = i == 0
        is_last = i == n_layers - 1
        layer_x_scale = x_scale if is_first else 1.0
        layer_x_zp = x_zero_point if is_first else 0
        layer_x_np_dtype = x_np_dtype if is_first else np.int8

        layer_y_scale = y_scales[i]
        layer_y_zp = last_y_zero_point if is_last else 0
        layer_y_dtype = last_y_dtype if is_last else TensorProto.INT8
        layer_y_np_dtype = (np.int8 if layer_y_dtype == TensorProto.INT8
                            else np.uint8)

        w_scale_arr = np.asarray(w_scales[i], dtype=np.float32)
        if w_scale_arr.ndim == 0:
            w_scale_arr = w_scale_arr.reshape(())

        p = f"L{i}_"
        initializers += [
            numpy_helper.from_array(w, name=p + "W"),
            numpy_helper.from_array(
                np.array(layer_x_scale, dtype=np.float32), name=p + "xs"),
            numpy_helper.from_array(
                np.array(layer_x_zp, dtype=layer_x_np_dtype),
                name=p + "xzp"),
            numpy_helper.from_array(w_scale_arr, name=p + "ws"),
            numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=p + "wzp"),
            numpy_helper.from_array(
                np.array(layer_y_scale, dtype=np.float32), name=p + "ys"),
            numpy_helper.from_array(
                np.array(layer_y_zp, dtype=layer_y_np_dtype),
                name=p + "yzp"),
        ]
        node_inputs = [cur_name, p + "xs", p + "xzp", p + "W", p + "ws",
                      p + "wzp", p + "ys", p + "yzp"]
        if is_last and last_bias is not None:
            initializers.append(numpy_helper.from_array(
                np.asarray(last_bias, dtype=np.int32), name=p + "B"))
            node_inputs.append(p + "B")
        out_name = f"L{i}_Y"
        nodes.append(helper.make_node(
            "QLinearConv", node_inputs, [out_name], name=f"conv{i}",
            strides=[stride_h, stride_w],
            pads=[pad_top, pad_left, pad_bottom, pad_right],
            dilations=[dilation_h, dilation_w]))

        cur_name = out_name
        cur_h, cur_w = out_h, out_w

    last_cout = weights[-1].shape[0]
    graph = helper.make_graph(
        nodes, "conv_chain",
        [helper.make_tensor_value_info(x_name, first_x_dtype, list(x_shape))],
        [helper.make_tensor_value_info(
            cur_name, layer_y_dtype, [n, last_cout, cur_h, cur_w])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    if check:
        onnx.checker.check_model(model)
    return model


def conv_pool_chain_model(weights, x_shape, pools, x_scale=1.0,
                          x_zero_point=0, x_dtype=TensorProto.INT8,
                          w_scales=None, y_scales=None, strides=None,
                          pads=None, dilations=None, last_bias=None,
                          last_y_zero_point=0, last_y_dtype=TensorProto.INT8,
                          check=True):
    """`conv_chain_model`, with a MaxPool optionally sitting between any
    two consecutive QLinearConv layers -- the way
    cim_frontend.onnx_import.load_conv_pool_chain reads (see its own
    module section header). Every parameter but `pools` is EXACTLY
    `conv_chain_model`'s own (see its docstring); this function is built
    by inserting one optional MaxPool node into that function's own
    per-layer loop, not by reimplementing it, so the two cannot silently
    drift apart on what a "layer" looks like.

    `pools` is a `{bridge_index: dict}` mapping -- bridge_index i means
    "a MaxPool sits between layer i's own Y and layer i+1's own X",
    exactly `cim_frontend.emit.emit_conv_chain_module`'s own
    `pool_params` bridge-index convention. Each dict may set
    `kernel_shape` (required, a `(kh, kw)` pair), `strides` (default
    `(2, 2)` -- NOT `(1, 1)`, since load_conv_pool_chain refuses
    `strides == 1` outright, see its own module header; a test that wants
    to exercise that refusal passes `strides=(1, 1)` explicitly), `pads`
    (default `(0, 0, 0, 0)`), `dilations` (default `(1, 1)`), and
    `ceil_mode` (default 0).
    """
    weights = [np.asarray(w) for w in weights]
    n_layers = len(weights)
    if n_layers < 2:
        raise ValueError("conv_pool_chain_model needs at least two conv "
                         "layers")
    for i in pools:
        if not (0 <= i < n_layers - 1):
            raise ValueError(
                f"pools has an entry for bridge {i}, which is out of "
                f"range for {n_layers} layers (0 <= bridge < "
                f"{n_layers - 1})")
    if w_scales is None:
        w_scales = [1.0] * n_layers
    if y_scales is None:
        y_scales = [1.0] * n_layers
    if strides is None:
        strides = [(1, 1)] * n_layers
    if pads is None:
        pads = [(0, 0, 0, 0)] * n_layers
    if dilations is None:
        dilations = [(1, 1)] * n_layers
    for name, seq in (("w_scales", w_scales), ("y_scales", y_scales),
                      ("strides", strides), ("pads", pads),
                      ("dilations", dilations)):
        if len(seq) != n_layers:
            raise ValueError(f"{name} must have one entry per layer "
                             f"({n_layers}), got {len(seq)}")

    n, cin0, h0, w0 = x_shape
    if weights[0].shape[1] != cin0:
        raise ValueError(
            f"x_shape's Cin ({cin0}) does not match layer 0's own Cin "
            f"({weights[0].shape[1]})")
    for i in range(1, n_layers):
        if weights[i].shape[1] != weights[i - 1].shape[0]:
            raise ValueError(
                f"layer {i}'s Cin ({weights[i].shape[1]}) does not match "
                f"layer {i - 1}'s Cout ({weights[i - 1].shape[0]})")

    x_name = "X"
    x_np_dtype = np.int8 if x_dtype == TensorProto.INT8 else np.uint8
    nodes = []
    initializers = []
    cur_name = x_name
    cur_h, cur_w = h0, w0
    first_x_dtype = x_dtype
    layer_y_dtype = TensorProto.INT8

    for i, w in enumerate(weights):
        cout, cin, kh, kw = w.shape
        stride_h, stride_w = strides[i]
        pad_top, pad_left, pad_bottom, pad_right = pads[i]
        dilation_h, dilation_w = dilations[i]
        dilated_kh = (kh - 1) * dilation_h + 1
        dilated_kw = (kw - 1) * dilation_w + 1
        out_h = (cur_h + pad_top + pad_bottom - dilated_kh) // stride_h + 1
        out_w = (cur_w + pad_left + pad_right - dilated_kw) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"layer {i}: non-positive output size ({out_h}, {out_w}) "
                f"for the given x_shape/weights/strides/pads/dilations")

        is_first = i == 0
        is_last = i == n_layers - 1
        layer_x_scale = x_scale if is_first else 1.0
        layer_x_zp = x_zero_point if is_first else 0
        layer_x_np_dtype = x_np_dtype if is_first else np.int8

        layer_y_scale = y_scales[i]
        layer_y_zp = last_y_zero_point if is_last else 0
        layer_y_dtype = last_y_dtype if is_last else TensorProto.INT8
        layer_y_np_dtype = (np.int8 if layer_y_dtype == TensorProto.INT8
                            else np.uint8)

        w_scale_arr = np.asarray(w_scales[i], dtype=np.float32)
        if w_scale_arr.ndim == 0:
            w_scale_arr = w_scale_arr.reshape(())

        p = f"L{i}_"
        initializers += [
            numpy_helper.from_array(w, name=p + "W"),
            numpy_helper.from_array(
                np.array(layer_x_scale, dtype=np.float32), name=p + "xs"),
            numpy_helper.from_array(
                np.array(layer_x_zp, dtype=layer_x_np_dtype),
                name=p + "xzp"),
            numpy_helper.from_array(w_scale_arr, name=p + "ws"),
            numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=p + "wzp"),
            numpy_helper.from_array(
                np.array(layer_y_scale, dtype=np.float32), name=p + "ys"),
            numpy_helper.from_array(
                np.array(layer_y_zp, dtype=layer_y_np_dtype),
                name=p + "yzp"),
        ]
        node_inputs = [cur_name, p + "xs", p + "xzp", p + "W", p + "ws",
                      p + "wzp", p + "ys", p + "yzp"]
        if is_last and last_bias is not None:
            initializers.append(numpy_helper.from_array(
                np.asarray(last_bias, dtype=np.int32), name=p + "B"))
            node_inputs.append(p + "B")
        out_name = f"L{i}_Y"
        nodes.append(helper.make_node(
            "QLinearConv", node_inputs, [out_name], name=f"conv{i}",
            strides=[stride_h, stride_w],
            pads=[pad_top, pad_left, pad_bottom, pad_right],
            dilations=[dilation_h, dilation_w]))

        cur_name = out_name
        cur_h, cur_w = out_h, out_w

        if i in pools:
            pool_kwargs = pools[i]
            pkh, pkw = pool_kwargs["kernel_shape"]
            p_stride_h, p_stride_w = pool_kwargs.get("strides", (2, 2))
            p_pad_top, p_pad_left, p_pad_bottom, p_pad_right = (
                pool_kwargs.get("pads", (0, 0, 0, 0)))
            p_dilation_h, p_dilation_w = pool_kwargs.get(
                "dilations", (1, 1))
            ceil_mode = pool_kwargs.get("ceil_mode", 0)
            p_dilated_kh = (pkh - 1) * p_dilation_h + 1
            p_dilated_kw = (pkw - 1) * p_dilation_w + 1
            pool_out_h = ((cur_h + p_pad_top + p_pad_bottom - p_dilated_kh)
                         // p_stride_h + 1)
            pool_out_w = ((cur_w + p_pad_left + p_pad_right - p_dilated_kw)
                         // p_stride_w + 1)
            if pool_out_h <= 0 or pool_out_w <= 0:
                raise ValueError(
                    f"bridge {i}'s pool: non-positive output size "
                    f"({pool_out_h}, {pool_out_w})")
            pool_out_name = f"L{i}_pool"
            pool_kwargs_onnx = dict(
                kernel_shape=[pkh, pkw], strides=[p_stride_h, p_stride_w],
                pads=[p_pad_top, p_pad_left, p_pad_bottom, p_pad_right],
                dilations=[p_dilation_h, p_dilation_w])
            if ceil_mode:
                pool_kwargs_onnx["ceil_mode"] = ceil_mode
            nodes.append(helper.make_node(
                "MaxPool", [cur_name], [pool_out_name], name=f"pool{i}",
                **pool_kwargs_onnx))
            cur_name = pool_out_name
            cur_h, cur_w = pool_out_h, pool_out_w

    last_cout = weights[-1].shape[0]
    graph = helper.make_graph(
        nodes, "conv_pool_chain",
        [helper.make_tensor_value_info(x_name, first_x_dtype, list(x_shape))],
        [helper.make_tensor_value_info(
            cur_name, layer_y_dtype, [n, last_cout, cur_h, cur_w])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    if check:
        onnx.checker.check_model(model)
    return model


def conv_chain_matmul_chain_model(conv_weights, x_shape, matmul_weights,
                                  x_scale=1.0, x_zero_point=0,
                                  x_dtype=TensorProto.INT8,
                                  conv_w_scales=None, conv_y_scales=None,
                                  strides=None, pads=None, dilations=None,
                                  bridge_scales=None, check=True):
    """Two or more QLinearConv layers (connected directly, conv_chain_
    model's own convention) feeding one or more MatMulInteger layers via
    the SAME Transpose(perm=[0, 2, 3, 1]) -> Reshape([M, Cout]) bridge
    conv_matmul_chain_model uses between a SINGLE conv and its first
    matmul -- the way cim_frontend.onnx_import.load_conv_chain_matmul_
    chain reads (PR D of the "chain convolution layers" plan, a real
    conv-stem-feeding-an-FC-head chain).

    `conv_weights[i]` is layer i's own [Cout_i, Cin_i, Kh_i, Kw_i] weight
    (ONNX layout), with Cin_i == Cout_(i-1) for i > 0.
    `conv_w_scales`/`conv_y_scales`/`strides`/`pads`/`dilations` are layer
    i's own, defaulting to 1.0/1.0/(1, 1)/(0, 0, 0, 0)/(1, 1) for every
    conv layer if not given -- EVERY conv layer here (including the
    last) keeps y_zero_point=0 and a scalar w_scale; unlike
    conv_chain_model's own last-layer flexibility, no conv layer here is
    ever the graph's own output (see load_conv_chain_matmul_chain's own
    module header for why).

    `matmul_weights` is a list of `[K_i, N_i]` arrays (ONNX layout, as
    matmul_chain_model/conv_matmul_chain_model take), with `K_0 ==` the
    LAST conv layer's own Cout. `bridge_scales`, if given, is a list of
    `len(matmul_weights) - 1` floats for the bridges BETWEEN matmul
    layers (never the conv-chain's own internal bridges, nor the
    conv-to-matmul bridge -- those are always derived from each
    conv layer's own `y_scale / (x_scale * w_scale)`, exactly
    conv_matmul_chain_model's own convention).
    """
    conv_weights = [np.asarray(w) for w in conv_weights]
    n_conv = len(conv_weights)
    if n_conv < 2:
        raise ValueError("conv_chain_matmul_chain_model needs at least "
                         "two conv layers")
    if conv_w_scales is None:
        conv_w_scales = [1.0] * n_conv
    if conv_y_scales is None:
        conv_y_scales = [1.0] * n_conv
    if strides is None:
        strides = [(1, 1)] * n_conv
    if pads is None:
        pads = [(0, 0, 0, 0)] * n_conv
    if dilations is None:
        dilations = [(1, 1)] * n_conv
    for name, seq in (("conv_w_scales", conv_w_scales),
                      ("conv_y_scales", conv_y_scales),
                      ("strides", strides), ("pads", pads),
                      ("dilations", dilations)):
        if len(seq) != n_conv:
            raise ValueError(f"{name} must have one entry per conv layer "
                             f"({n_conv}), got {len(seq)}")

    matmul_weights = [np.asarray(w) for w in matmul_weights]
    if matmul_weights[0].shape[0] != conv_weights[-1].shape[0]:
        raise ValueError(
            f"matmul_weights[0]'s K ({matmul_weights[0].shape[0]}) does "
            f"not match the last conv layer's own Cout "
            f"({conv_weights[-1].shape[0]})")
    if bridge_scales is None:
        bridge_scales = [1.0] * (len(matmul_weights) - 1)
    if len(bridge_scales) != len(matmul_weights) - 1:
        raise ValueError(
            f"bridge_scales must have one entry per inter-matmul bridge "
            f"(len(matmul_weights) - 1 = {len(matmul_weights) - 1}), got "
            f"{len(bridge_scales)}")

    n, cin0, h0, w0 = x_shape
    if conv_weights[0].shape[1] != cin0:
        raise ValueError(
            f"x_shape's Cin ({cin0}) does not match layer 0's own Cin "
            f"({conv_weights[0].shape[1]})")
    for i in range(1, n_conv):
        if conv_weights[i].shape[1] != conv_weights[i - 1].shape[0]:
            raise ValueError(
                f"layer {i}'s Cin ({conv_weights[i].shape[1]}) does not "
                f"match layer {i - 1}'s Cout ({conv_weights[i - 1].shape[0]})")

    x_name = "X"
    x_np_dtype = np.int8 if x_dtype == TensorProto.INT8 else np.uint8
    initializers = []
    nodes = []
    cur_name = x_name
    cur_h, cur_w = h0, w0
    last_cout = None

    for i, w in enumerate(conv_weights):
        cout, cin, kh, kw = w.shape
        stride_h, stride_w = strides[i]
        pad_top, pad_left, pad_bottom, pad_right = pads[i]
        dilation_h, dilation_w = dilations[i]
        dilated_kh = (kh - 1) * dilation_h + 1
        dilated_kw = (kw - 1) * dilation_w + 1
        out_h = (cur_h + pad_top + pad_bottom - dilated_kh) // stride_h + 1
        out_w = (cur_w + pad_left + pad_right - dilated_kw) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"layer {i}: non-positive output size ({out_h}, {out_w})")

        is_first = i == 0
        layer_x_scale = x_scale if is_first else 1.0
        layer_x_zp = x_zero_point if is_first else 0
        layer_x_np_dtype = x_np_dtype if is_first else np.int8

        p = f"L{i}_"
        initializers += [
            numpy_helper.from_array(w, name=p + "W"),
            numpy_helper.from_array(
                np.array(layer_x_scale, dtype=np.float32), name=p + "xs"),
            numpy_helper.from_array(
                np.array(layer_x_zp, dtype=layer_x_np_dtype),
                name=p + "xzp"),
            numpy_helper.from_array(
                np.array(conv_w_scales[i], dtype=np.float32), name=p + "ws"),
            numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=p + "wzp"),
            numpy_helper.from_array(
                np.array(conv_y_scales[i], dtype=np.float32), name=p + "ys"),
            numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=p + "yzp"),
        ]
        out_name = f"L{i}_Y"
        nodes.append(helper.make_node(
            "QLinearConv",
            [cur_name, p + "xs", p + "xzp", p + "W", p + "ws", p + "wzp",
             p + "ys", p + "yzp"],
            [out_name], name=f"conv{i}", strides=[stride_h, stride_w],
            pads=[pad_top, pad_left, pad_bottom, pad_right],
            dilations=[dilation_h, dilation_w]))

        cur_name = out_name
        cur_h, cur_w = out_h, out_w
        last_cout = cout

    m = n * cur_h * cur_w
    transpose_node = helper.make_node(
        "Transpose", [cur_name], ["bridge_nhwc"], perm=[0, 2, 3, 1],
        name="bridge_transpose")
    initializers.append(numpy_helper.from_array(
        np.array([m, last_cout], dtype=np.int64), name="bridge_reshape_shape"))
    reshape_node = helper.make_node(
        "Reshape", ["bridge_nhwc", "bridge_reshape_shape"], ["bridge_flat"],
        name="bridge_reshape")

    nodes += [transpose_node, reshape_node]
    cur = "bridge_flat"
    for i, mw in enumerate(matmul_weights):
        w_name = f"mmW{i}"
        y_name = f"mmY{i}"
        initializers.append(numpy_helper.from_array(mw, name=w_name))
        nodes.append(helper.make_node("MatMulInteger", [cur, w_name],
                                      [y_name], name=f"mm{i}"))
        if i < len(matmul_weights) - 1:
            float_name, q_name = f"mmf{i}", f"mmq{i}"
            scale_name, zp_name = f"mmscale{i}", f"mmzp{i}"
            initializers.append(numpy_helper.from_array(
                np.array(bridge_scales[i], dtype=np.float32),
                name=scale_name))
            initializers.append(numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=zp_name))
            nodes.append(helper.make_node(
                "Cast", [y_name], [float_name], to=TensorProto.FLOAT,
                name=f"mmcast{i}"))
            nodes.append(helper.make_node(
                "QuantizeLinear", [float_name, scale_name, zp_name],
                [q_name], name=f"mmquant{i}"))
            cur = q_name

    n_last = matmul_weights[-1].shape[1]
    graph = helper.make_graph(
        nodes, "conv_chain_matmul_chain",
        [helper.make_tensor_value_info(x_name, x_dtype, list(x_shape))],
        [helper.make_tensor_value_info(
            f"mmY{len(matmul_weights) - 1}", TensorProto.INT32,
            [m, n_last])],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    if check:
        onnx.checker.check_model(model)
    return model


def conv_pool_chain_matmul_chain_model(conv_weights, x_shape, matmul_weights,
                                       pools=None, x_scale=1.0,
                                       x_zero_point=0,
                                       x_dtype=TensorProto.INT8,
                                       conv_w_scales=None, conv_y_scales=None,
                                       strides=None, pads=None,
                                       dilations=None, bridge_scales=None,
                                       check=True):
    """`conv_chain_matmul_chain_model`, with a MaxPool optionally sitting
    between any two consecutive conv layers OR between the chain's own
    last conv layer and the conv->matmul bridge -- the way
    cim_frontend.onnx_import.load_conv_pool_chain_matmul_chain reads (see
    its own module section header). Every parameter but `pools` is
    EXACTLY `conv_chain_matmul_chain_model`'s own; this function is built
    by inserting one optional MaxPool node into that function's own
    per-layer loop, not by reimplementing it, the same relationship
    `conv_pool_chain_model` has to `conv_chain_model`.

    `pools` is a `{bridge_index: dict}` mapping, exactly
    `conv_pool_chain_model`'s own convention, EXCEPT `bridge_index ==
    n_conv - 1` (a pool between the last conv and the conv->matmul
    bridge) is valid here -- the one position `conv_pool_chain_model`
    itself refuses (it has no bridge for such a pool to feed into).
    """
    conv_weights = [np.asarray(w) for w in conv_weights]
    n_conv = len(conv_weights)
    if n_conv < 2:
        raise ValueError("conv_pool_chain_matmul_chain_model needs at "
                         "least two conv layers")
    if pools is None:
        pools = {}
    for i in pools:
        if not (0 <= i < n_conv):
            raise ValueError(
                f"pools has an entry for bridge {i}, which is out of "
                f"range for {n_conv} conv layers (0 <= bridge < {n_conv})")
    if conv_w_scales is None:
        conv_w_scales = [1.0] * n_conv
    if conv_y_scales is None:
        conv_y_scales = [1.0] * n_conv
    if strides is None:
        strides = [(1, 1)] * n_conv
    if pads is None:
        pads = [(0, 0, 0, 0)] * n_conv
    if dilations is None:
        dilations = [(1, 1)] * n_conv
    for name, seq in (("conv_w_scales", conv_w_scales),
                      ("conv_y_scales", conv_y_scales),
                      ("strides", strides), ("pads", pads),
                      ("dilations", dilations)):
        if len(seq) != n_conv:
            raise ValueError(f"{name} must have one entry per conv layer "
                             f"({n_conv}), got {len(seq)}")

    matmul_weights = [np.asarray(w) for w in matmul_weights]
    if bridge_scales is None:
        bridge_scales = [1.0] * (len(matmul_weights) - 1)
    if len(bridge_scales) != len(matmul_weights) - 1:
        raise ValueError(
            f"bridge_scales must have one entry per inter-matmul bridge "
            f"(len(matmul_weights) - 1 = {len(matmul_weights) - 1}), got "
            f"{len(bridge_scales)}")

    n, cin0, h0, w0 = x_shape
    if conv_weights[0].shape[1] != cin0:
        raise ValueError(
            f"x_shape's Cin ({cin0}) does not match layer 0's own Cin "
            f"({conv_weights[0].shape[1]})")
    for i in range(1, n_conv):
        if conv_weights[i].shape[1] != conv_weights[i - 1].shape[0]:
            raise ValueError(
                f"layer {i}'s Cin ({conv_weights[i].shape[1]}) does not "
                f"match layer {i - 1}'s Cout ({conv_weights[i - 1].shape[0]})")

    x_name = "X"
    x_np_dtype = np.int8 if x_dtype == TensorProto.INT8 else np.uint8
    initializers = []
    nodes = []
    cur_name = x_name
    cur_h, cur_w = h0, w0
    last_cout = None

    for i, w in enumerate(conv_weights):
        cout, cin, kh, kw = w.shape
        stride_h, stride_w = strides[i]
        pad_top, pad_left, pad_bottom, pad_right = pads[i]
        dilation_h, dilation_w = dilations[i]
        dilated_kh = (kh - 1) * dilation_h + 1
        dilated_kw = (kw - 1) * dilation_w + 1
        out_h = (cur_h + pad_top + pad_bottom - dilated_kh) // stride_h + 1
        out_w = (cur_w + pad_left + pad_right - dilated_kw) // stride_w + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"layer {i}: non-positive output size ({out_h}, {out_w})")

        is_first = i == 0
        layer_x_scale = x_scale if is_first else 1.0
        layer_x_zp = x_zero_point if is_first else 0
        layer_x_np_dtype = x_np_dtype if is_first else np.int8

        p = f"L{i}_"
        initializers += [
            numpy_helper.from_array(w, name=p + "W"),
            numpy_helper.from_array(
                np.array(layer_x_scale, dtype=np.float32), name=p + "xs"),
            numpy_helper.from_array(
                np.array(layer_x_zp, dtype=layer_x_np_dtype),
                name=p + "xzp"),
            numpy_helper.from_array(
                np.array(conv_w_scales[i], dtype=np.float32), name=p + "ws"),
            numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=p + "wzp"),
            numpy_helper.from_array(
                np.array(conv_y_scales[i], dtype=np.float32), name=p + "ys"),
            numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=p + "yzp"),
        ]
        out_name = f"L{i}_Y"
        nodes.append(helper.make_node(
            "QLinearConv",
            [cur_name, p + "xs", p + "xzp", p + "W", p + "ws", p + "wzp",
             p + "ys", p + "yzp"],
            [out_name], name=f"conv{i}", strides=[stride_h, stride_w],
            pads=[pad_top, pad_left, pad_bottom, pad_right],
            dilations=[dilation_h, dilation_w]))

        cur_name = out_name
        cur_h, cur_w = out_h, out_w
        last_cout = cout

        if i in pools:
            pool_kwargs = pools[i]
            pkh, pkw = pool_kwargs["kernel_shape"]
            p_stride_h, p_stride_w = pool_kwargs.get("strides", (2, 2))
            p_pad_top, p_pad_left, p_pad_bottom, p_pad_right = (
                pool_kwargs.get("pads", (0, 0, 0, 0)))
            p_dilation_h, p_dilation_w = pool_kwargs.get(
                "dilations", (1, 1))
            ceil_mode = pool_kwargs.get("ceil_mode", 0)
            p_dilated_kh = (pkh - 1) * p_dilation_h + 1
            p_dilated_kw = (pkw - 1) * p_dilation_w + 1
            pool_out_h = ((cur_h + p_pad_top + p_pad_bottom - p_dilated_kh)
                         // p_stride_h + 1)
            pool_out_w = ((cur_w + p_pad_left + p_pad_right - p_dilated_kw)
                         // p_stride_w + 1)
            if pool_out_h <= 0 or pool_out_w <= 0:
                raise ValueError(
                    f"bridge {i}'s pool: non-positive output size "
                    f"({pool_out_h}, {pool_out_w})")
            pool_out_name = f"L{i}_pool"
            pool_kwargs_onnx = dict(
                kernel_shape=[pkh, pkw], strides=[p_stride_h, p_stride_w],
                pads=[p_pad_top, p_pad_left, p_pad_bottom, p_pad_right],
                dilations=[p_dilation_h, p_dilation_w])
            if ceil_mode:
                pool_kwargs_onnx["ceil_mode"] = ceil_mode
            nodes.append(helper.make_node(
                "MaxPool", [cur_name], [pool_out_name], name=f"pool{i}",
                **pool_kwargs_onnx))
            cur_name = pool_out_name
            cur_h, cur_w = pool_out_h, pool_out_w

    m = n * cur_h * cur_w
    transpose_node = helper.make_node(
        "Transpose", [cur_name], ["bridge_nhwc"], perm=[0, 2, 3, 1],
        name="bridge_transpose")
    initializers.append(numpy_helper.from_array(
        np.array([m, last_cout], dtype=np.int64), name="bridge_reshape_shape"))
    reshape_node = helper.make_node(
        "Reshape", ["bridge_nhwc", "bridge_reshape_shape"], ["bridge_flat"],
        name="bridge_reshape")

    nodes += [transpose_node, reshape_node]
    if matmul_weights[0].shape[0] != last_cout:
        raise ValueError(
            f"matmul_weights[0]'s K ({matmul_weights[0].shape[0]}) does "
            f"not match the chain's own final Cout after any trailing "
            f"pool ({last_cout})")
    cur = "bridge_flat"
    for i, mw in enumerate(matmul_weights):
        w_name = f"mmW{i}"
        y_name = f"mmY{i}"
        initializers.append(numpy_helper.from_array(mw, name=w_name))
        nodes.append(helper.make_node("MatMulInteger", [cur, w_name],
                                      [y_name], name=f"mm{i}"))
        if i < len(matmul_weights) - 1:
            float_name, q_name = f"mmf{i}", f"mmq{i}"
            scale_name, zp_name = f"mmscale{i}", f"mmzp{i}"
            initializers.append(numpy_helper.from_array(
                np.array(bridge_scales[i], dtype=np.float32),
                name=scale_name))
            initializers.append(numpy_helper.from_array(
                np.array(0, dtype=np.int8), name=zp_name))
            nodes.append(helper.make_node(
                "Cast", [y_name], [float_name], to=TensorProto.FLOAT,
                name=f"mmcast{i}"))
            nodes.append(helper.make_node(
                "QuantizeLinear", [float_name, scale_name, zp_name],
                [q_name], name=f"mmquant{i}"))
            cur = q_name

    n_last = matmul_weights[-1].shape[1]
    graph = helper.make_graph(
        nodes, "conv_pool_chain_matmul_chain",
        [helper.make_tensor_value_info(x_name, x_dtype, list(x_shape))],
        [helper.make_tensor_value_info(
            f"mmY{len(matmul_weights) - 1}", TensorProto.INT32,
            [m, n_last])],
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
