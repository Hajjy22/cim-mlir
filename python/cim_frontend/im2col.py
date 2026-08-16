"""im2col: turn a 2-D convolution's activation into a matmul's activation.

Deliberately free of any `onnx` dependency, same reason as emit.py: this is
pure numpy array shuffling, testable and reviewable without the model-
reading library installed at all. onnx_import.py is the only place that
knows what a QLinearConv node is.

WHY THIS EXISTS AT ALL
=======================
The cim pipeline has exactly one primitive that touches a weight-stationary
tile: a matrix-vector (or, since M > 1 batching landed, matrix-matrix)
multiply. A convolution is not that op in general -- but THIS project's
convolution is not "general": onnx_import.py's own module docstring already
establishes that every emitted module is ONE BAKED INFERENCE, because
cim-run takes no runtime inputs and the activation is always a compile-time
constant. That single fact removes the usual reason im2col is considered
expensive (materializing a patches matrix at every inference, redundantly
storing overlapping windows): here it happens exactly once, at import time,
in Python, before any MLIR is even emitted. So a 2-D convolution over a
constant activation and a constant kernel is *exactly* a matmul in
disguise: reshape the kernel to [Cout, Cin*Kh*Kw] (already output-major,
see below), im2col the activation to [N*OutH*OutW, Cin*Kh*Kw], and hand
both to the SAME emit.emit_module this project already runs a full
verification gate against for the plain matmul case. No new MLIR op, no
interpreter change, no cim-detect/cim-partition change -- the entire
feature is a front-end reshape.

THE KERNEL NEEDS NO TRANSPOSE (unlike MatMulInteger's B)
=========================================================
onnx_import.py's own "THE TRANSPOSE" note explains why MatMulInteger's
weight (ONNX layout [K, N]) has to be flipped to cim.mvm's output-major
[N, K] before it can be used. A conv kernel does not have this problem:
ONNX's Conv/QLinearConv weight is already [Cout, Cin, Kh, Kw] -- Cout-major
-- so `weight.reshape(Cout, -1)` is ALREADY in [N, K] form with no flip
needed. This module's im2col output is built to match that flatten order
exactly (see `im2col_nchw`'s docstring) so the two reshapes line up without
a transpose appearing anywhere in this file.

WHAT THIS DOES NOT HANDLE
==========================
Grouped convolution: each group needs its own, disjoint matmul over a
slice of channels, not expressible as a single reshape the way dilation
(below) is. onnx_import.py refuses it explicitly rather than silently
computing plain convolution's answer for a grouped (e.g. depthwise) model.

DILATION IS A SAMPLING PATTERN, NOT A SHAPE CHANGE
====================================================
A dilated kernel tap reads a non-contiguous patch, but "non-contiguous"
only means the SOURCE indices are strided -- the DESTINATION patch this
function builds is still a dense [C, Kh, Kw] block per output position,
identical in shape to the non-dilated case. numpy's own strided slicing
(`start:stop:step`) reads exactly that pattern directly, with the step set
to the dilation factor, so no different data structure or a second pass is
needed: `im2col_nchw` below takes `dilation_h`/`dilation_w` (default 1,
identical to every call site before dilation was supported) and the same
loop already here does the rest. Contrast with grouped convolution above,
which genuinely cannot be expressed this way.
"""

import numpy as np


def _pad_nchw(x, pad_top, pad_bottom, pad_left, pad_right):
    """Zero-pad the H and W axes of an [N, C, H, W] array.

    Zero is the correct pad value ONLY because onnx_import.py requires
    x_zero_point == 0 before ever calling this: in a symmetric-int8
    quantized tensor, the quantized value 0 IS the logical zero a
    convolution's implicit padding is supposed to contribute. A non-zero
    zero point would need the pad value to be x_zero_point instead, which
    is exactly the case onnx_import.py's own refusal exists to avoid ever
    reaching this function with.
    """
    if pad_top == pad_bottom == pad_left == pad_right == 0:
        return x
    n, c, h, w = x.shape
    padded = np.zeros(
        (n, c, h + pad_top + pad_bottom, w + pad_left + pad_right),
        dtype=x.dtype)
    padded[:, :, pad_top:pad_top + h, pad_left:pad_left + w] = x
    return padded


def im2col_nchw(x, kh, kw, stride_h, stride_w,
                pad_top, pad_bottom, pad_left, pad_right,
                dilation_h=1, dilation_w=1):
    """[N, C, H, W] -> ([N*OutH*OutW, C*Kh*Kw] patches, (N, OutH, OutW)).

    Row `m` of the returned patches matrix is output position
    `(n, oh, ow) = np.unravel_index(m, (N, OutH, OutW))`'s receptive field,
    flattened in (C, Kh, Kw) order -- the SAME order `weight.reshape(Cout,
    -1)` flattens a [Cout, C, Kh, Kw] kernel in, by construction (both are
    numpy's default C-order flatten of the same three trailing axes), so
    `patches @ weight.reshape(Cout, -1).T` is exactly the convolution's raw
    (pre-quantization) accumulator, in `linalg.matmul_transpose_b`'s own
    ins/outs convention.

    `dilation_h`/`dilation_w` (default 1, ordinary convolution) space the
    Kh*Kw taps `dilation` elements apart instead of adjacent -- see this
    module's own "DILATION IS A SAMPLING PATTERN" note. The output-size
    formula matches the ONNX spec's own
    `floor((input + pad - dilation*(kernel-1) - 1) / stride) + 1`
    algebraically: `dilated_kh = (kh-1)*dilation_h + 1` is the span an
    already-dilated kernel covers, so `(hp - dilated_kh)/stride + 1`
    is the same expression rearranged around that span instead of the
    kernel size directly.

    Built as an explicit loop over output positions rather than a strided-
    view trick: this is import-time, one-shot, non-performance-critical
    code, and a loop whose body is one slice assignment is straightforward
    to read against the definition of a convolution, which is what a
    numerical front end most needs to be.
    """
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"im2col_nchw needs a rank-4 [N, C, H, W] array, "
                         f"got shape {x.shape}")
    n, c, h, w = x.shape
    if kh <= 0 or kw <= 0:
        raise ValueError(f"kernel size must be positive, got ({kh}, {kw})")
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError(
            f"stride must be positive, got ({stride_h}, {stride_w})")
    if dilation_h <= 0 or dilation_w <= 0:
        raise ValueError(
            f"dilation must be positive, got ({dilation_h}, {dilation_w})")

    xp = _pad_nchw(x, pad_top, pad_bottom, pad_left, pad_right)
    hp, wp = xp.shape[2], xp.shape[3]
    dilated_kh = (kh - 1) * dilation_h + 1
    dilated_kw = (kw - 1) * dilation_w + 1
    if dilated_kh > hp or dilated_kw > wp:
        raise ValueError(
            f"dilated kernel ({dilated_kh}, {dilated_kw}) -- kernel "
            f"({kh}, {kw}) at dilation ({dilation_h}, {dilation_w}) -- is "
            f"larger than the padded input ({hp}, {wp})")

    out_h = (hp - dilated_kh) // stride_h + 1
    out_w = (wp - dilated_kw) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"non-positive output size ({out_h}, {out_w}) for input "
            f"{(h, w)}, kernel ({kh}, {kw}), stride ({stride_h}, "
            f"{stride_w}), pad ({pad_top}, {pad_bottom}, {pad_left}, "
            f"{pad_right}), dilation ({dilation_h}, {dilation_w})")

    patches = np.empty((n, out_h, out_w, c, kh, kw), dtype=xp.dtype)
    for oh in range(out_h):
        h0 = oh * stride_h
        for ow in range(out_w):
            w0 = ow * stride_w
            patches[:, oh, ow] = xp[:, :,
                                    h0:h0 + dilated_kh:dilation_h,
                                    w0:w0 + dilated_kw:dilation_w]

    return patches.reshape(n * out_h * out_w, c * kh * kw), (n, out_h, out_w)
