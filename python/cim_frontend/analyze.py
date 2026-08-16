"""Permissive graph walk: offloadable layer shapes for cost analysis.

`onnx_import.py`'s loaders (`load_matmul_integer`, `load_matmul_chain`,
`load_qlinear_conv`) exist to produce MLIR that actually runs, so each one
refuses the ENTIRE graph the moment it contains one op it does not
recognize -- the only safe behavior when silently dropping an op would
change the function being compiled (see refusal.py's own module
docstring). That is the right rule for compilation. It is the wrong rule
for a question this project's placement/cost engine can actually answer
without ever executing anything: "how much weight-residency pressure does
this real network put on this chip?"

THE INSIGHT THIS MODULE ACTS ON
================================
Placement analysis needs only a layer's weight SHAPE -- the [K, N] pair
`cim.Placement.Workloads.partitionBlockCount` turns into a tile-block
count -- never its values, and never an activation. A `MaxPool` or
`Softmax` node has no resident weights and so cannot change that count no
matter what it computes; a grouped `QLinearConv` this project's compiler
cannot yet EMIT still has a real, known weight tensor whose shape this
module can still report. So this walker never refuses the graph:
every node is either an offloadable layer (its shape goes in `layers`) or
a skip (its op type and the reason go in `skipped`), and the walk always
completes.

Nodes are visited in `graph.node` order, which the ONNX spec already
requires to be topologically sorted -- there is no additional ordering
work to do here.

THE HONESTY REQUIREMENT (non-negotiable; see docs/roadmap.md)
===============================================================
A report that counts only the layers it understood, without saying how
many others existed and why they were left out, is exactly the kind of
confident-but-partial number this project refuses to publish elsewhere.
Every call returns BOTH `layers` and `skipped`, plus a `note` stating in
words that the cost this feeds is weight-programming cost for the N
offloadable layers only, not end-to-end inference cost. `cim-bench
analyze` (the C++ consumer) is expected to repeat this disclosure rather
than silently drop it once it has what it needs.

WHY THIS DUPLICATES A FEW LINES OF onnx_import.py RATHER THAN CALLING IT
==========================================================================
`load_matmul_integer`/`load_qlinear_conv` mix three kinds of check:
graph-scope ("is this the only node of this type, and is every other node
recognized"), activation-dependent (shape/dtype checks against a supplied
runtime input), and per-operand/weight checks that need neither. Only the
third kind applies here -- this walker has no activation and explicitly
tolerates other node types -- so the two per-node extractors below inline
that subset directly, reusing the field-level validators
(`_zero_point_is_zero`, `_positive_weight_scale`, `_scalar_zero_point`,
`_weight_zero_point_must_be_zero`) from `onnx_import` unchanged rather
than duplicating THEIR logic. Refusal wording is kept close to the
originals on purpose: the same defect should read as the same sentence
whether it blocked compilation or was merely skipped here.
"""

import numpy as np

from .onnx_import import (ACCEPTED_CONV_OP, ACCEPTED_OP, _check_opset,
                          _positive_weight_scale, _scalar_zero_point,
                          _tensor_names, _weight_zero_point_must_be_zero,
                          _zero_point_is_zero)
from .refusal import Refusal

_NOTHING_EMITTED_SUFFIX = " Nothing emitted."


def _reason(refusal):
    """A Refusal's message, minus the compile-path-specific "Nothing
    emitted." suffix -- this walker does not abort on one node's refusal,
    so that suffix would be misleading here."""
    msg = refusal.message
    if msg.endswith(_NOTHING_EMITTED_SUFFIX):
        msg = msg[: -len(_NOTHING_EMITTED_SUFFIX)]
    return msg


def _matmul_integer_kn(model, initializers, node):
    """(k, n) for a MatMulInteger node's weight, [K, N] as ONNX itself
    lays it out -- no transpose, unlike load_matmul_integer's return value,
    since partitionBlockCount only needs the pair, not cim.mvm's output-
    major convention. Raises Refusal for anything load_matmul_integer
    would also refuse."""
    from onnx import numpy_helper

    graph = model.graph
    where = f"node '{node.name or ACCEPTED_OP}'"
    if len(node.input) < 2:
        raise Refusal(
            f"{ACCEPTED_OP} has {len(node.input)} input(s); A and B are "
            f"both mandatory.", where=where)
    a_name, b_name = node.input[0], node.input[1]

    if b_name not in initializers:
        raise Refusal(
            f"the weight operand '{b_name}' is not a constant initializer. "
            f"Weight-stationary hardware has nothing to make resident if "
            f"the weights are computed at runtime.", where=where)
    if a_name in initializers:
        raise Refusal(
            "both operands are constant initializers; a matmul of two "
            "constants has nothing to make resident.", where=where)

    a_zp = node.input[2] if len(node.input) > 2 else ""
    b_zp = node.input[3] if len(node.input) > 3 else ""
    _zero_point_is_zero(model, a_zp, node.name or ACCEPTED_OP, "a_zero_point")
    _zero_point_is_zero(model, b_zp, node.name or ACCEPTED_OP, "b_zero_point")

    b_init = next(i for i in graph.initializer if i.name == b_name)
    weight = numpy_helper.to_array(b_init)
    if weight.dtype != np.int8:
        raise Refusal(
            f"the weight operand '{b_name}' has dtype {weight.dtype}; "
            f"cim.mvm and the simulator are signed throughout.",
            where=where)
    if weight.ndim != 2:
        raise Refusal(
            f"the weight operand '{b_name}' has rank {weight.ndim}; "
            f"cim-partition requires rank-2 operands.", where=where)

    k, n = weight.shape
    return int(k), int(n)


def _qlinear_conv_kn(graph, initializers, node):
    """(k, n) = (Cin*Kh*Kw, Cout) for a QLinearConv node's weight, matching
    im2col_nchw's own reshape (see im2col.py's "THE KERNEL NEEDS NO
    TRANSPOSE" note). Unlike load_qlinear_conv, this accepts group != 1:
    the weight tensor's shape is real and known even though this front
    end cannot yet EMIT IR for a grouped convolution -- see this module's
    own docstring. (Dilation needs no such carve-out: onnx_import.py's
    compile path accepts any positive dilation too, same as here.) Still
    refuses what would make the shape itself unknowable or the op not
    truly weight-stationary."""
    from onnx import numpy_helper

    where = f"node '{node.name or ACCEPTED_CONV_OP}'"
    if len(node.input) < 8:
        raise Refusal(
            f"{ACCEPTED_CONV_OP} has {len(node.input)} inputs; x, x_scale, "
            f"x_zero_point, w, w_scale, w_zero_point, y_scale and "
            f"y_zero_point are all mandatory per the ONNX spec.",
            where=where)
    (x_name, _x_scale_name, _x_zp_name, w_name, w_scale_name, w_zp_name,
     _y_scale_name, y_zp_name) = node.input[:8]

    if w_name not in initializers:
        raise Refusal(
            f"the weight operand '{w_name}' is not a constant initializer. "
            f"Weight-stationary hardware has nothing to make resident if "
            f"the weights are computed at runtime.", where=where)
    if x_name in initializers:
        raise Refusal(
            "both operands are constant initializers; nothing to make "
            "resident.", where=where)

    w_init = next(i for i in graph.initializer if i.name == w_name)
    weight = numpy_helper.to_array(w_init)
    if weight.dtype != np.int8:
        raise Refusal(
            f"the weight operand '{w_name}' has dtype {weight.dtype}.",
            where=where)
    if weight.ndim != 4:
        raise Refusal(
            f"the weight operand '{w_name}' has rank {weight.ndim}; a 2-D "
            f"convolution's kernel is rank 4 [Cout, Cin, Kh, Kw].",
            where=where)
    cout, cin, kh, kw = weight.shape

    # Real checks, not shape-only: an unrecognized w_scale/zero_point shape
    # means this is not the symmetric-weight, per-tensor-or-per-channel
    # contract this project's cost model was calibrated against at all, so
    # a reported k/n would not describe a layer this project actually
    # understands -- as opposed to group below, which the weight tensor's
    # shape already accounts for regardless.
    _positive_weight_scale(graph, w_scale_name, node.name, "w_scale", cout)
    _weight_zero_point_must_be_zero(graph, w_zp_name, node.name, cout)
    _y_zero_point, y_zp_dtype = _scalar_zero_point(
        graph, y_zp_name, node.name, "y_zero_point", allow_nonzero=True)
    if y_zp_dtype != np.int8:
        raise Refusal(
            f"y_zero_point is {y_zp_dtype}; cim.requantize's clamp is a "
            f"SIGNED effective_bits range, which cannot represent a uint8 "
            f"output's full [0, 255] span.", where=where)

    # ONNX stores a grouped conv's weight as [Cout, Cin/group, Kh, Kw] --
    # Cin here is already the real per-filter channel count, so k = Cin *
    # Kh * Kw is correct regardless of `group` (this front end cannot yet
    # emit IR to COMPILE a grouped convolution, see module docstring) or
    # `dilations` (which the compile path now accepts too, so there is no
    # divergence to explain there at all). Deliberately NOT read here at
    # all: an earlier version of this function called _int_attr/
    # _int_list_attr on them "to check that something read-able exists",
    # but neither helper can actually fail that check -- both read the
    # raw protobuf field for the requested type and silently return the
    # type's zero value if the attribute is stored under a different
    # AttributeProto type than expected, rather than raising. A call that
    # cannot fail and whose result is discarded is not a check; it was
    # dead code dressed up as one.
    return int(cin * kh * kw), int(cout)


def _has_subgraph(node):
    """True for a control-flow op (If/Loop/Scan/...) carrying a nested
    GraphProto -- one whose own nodes `graph.node` never visits, since
    ONNX stores a subgraph as an attribute payload, not as sibling nodes.
    Checked by attribute TYPE (GRAPH/GRAPHS), not by op name, so this
    covers any current or future op shaped that way, not just the three
    named here."""
    from onnx import AttributeProto

    return any(a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
              for a in node.attribute)


def analyze_model(model, model_path="<model>"):
    """Walk `model`'s graph, never refusing it as a whole.

    Returns a JSON-serializable dict:
        {"model": model_path,
         "layers": [{"name", "op_type", "k", "n"}, ...],
         "skipped": [{"name", "op_type", "reason"}, ...],
         "note": "<the honesty-requirement sentence, filled in>"}

    Still raises Refusal for a small number of GRAPH-level (not per-node)
    problems that make every node's classification untrustworthy -- today
    just an unusable opset import (`_check_opset`), the same check
    `import_model` itself runs first.

    LIMIT TO THE HONESTY REQUIREMENT: only the TOP-LEVEL graph is walked.
    A control-flow op (If/Loop/Scan) carries its branches as nested
    GraphProto attributes, not as sibling nodes `graph.node` iterates --
    any MatMulInteger/QLinearConv inside one is invisible to this walker,
    not merely uninteresting the way a MaxPool is. Silently treating that
    as "0 offloadable layers found there" would be exactly the confident-
    but-partial number this module exists to avoid, so such a node's own
    skip reason says its subgraph was never entered rather than reusing
    the generic "not a weight-stationary op" wording -- see
    `_has_subgraph`.
    """
    _check_opset(model)

    graph = model.graph
    initializers = _tensor_names(model)

    layers = []
    skipped = []
    for node in graph.node:
        label = node.name or f"<unnamed {node.op_type}>"
        try:
            if node.op_type == ACCEPTED_OP:
                k, n = _matmul_integer_kn(model, initializers, node)
            elif node.op_type == ACCEPTED_CONV_OP:
                k, n = _qlinear_conv_kn(graph, initializers, node)
            elif _has_subgraph(node):
                raise Refusal(
                    f"'{node.op_type}' carries a nested subgraph (e.g. an "
                    f"If/Loop/Scan branch) this walker does not enter; any "
                    f"{ACCEPTED_OP}/{ACCEPTED_CONV_OP} inside it is "
                    f"INVISIBLE to this report, not merely skipped -- it "
                    f"is neither counted in 'layers' nor named anywhere "
                    f"in 'skipped'.")
            else:
                raise Refusal(
                    f"'{node.op_type}' is not {ACCEPTED_OP} or "
                    f"{ACCEPTED_CONV_OP}, the only op types this project "
                    f"recognizes as weight-stationary; it has no resident "
                    f"weights to place on tiles.")
        except Refusal as refusal:
            skipped.append({"name": label, "op_type": node.op_type,
                            "reason": _reason(refusal)})
            continue
        layers.append({"name": label, "op_type": node.op_type, "k": k, "n": n})

    note = (
        f"weight-programming cost for the {len(layers)} offloadable "
        f"layer(s) listed in 'layers' ONLY -- this is NOT end-to-end "
        f"inference cost. {len(skipped)} other op(s) were skipped (see "
        f"'skipped' for which, and why); nothing they compute is "
        f"represented anywhere downstream of this report."
    )
    return {"model": model_path, "layers": layers, "skipped": skipped,
           "note": note}
