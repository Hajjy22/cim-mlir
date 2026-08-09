"""MLIR text emission for the ONNX front end.

Deliberately free of any `onnx` dependency: this module turns plain numpy
arrays into MLIR text, so it can be imported -- and tested -- in a build
that has no model-reading libraries installed at all. `onnx_import.py` is
the only place that knows what a model file is.

THE IR SHAPE HERE IS NOT FREE TO VARY. It must stay byte-identical, for
the single-layer case, to `build_module()` in
test/python/test_numerical_differential.py, which is in turn the same
shape as `buildModule()` in test/mlir/pipeline_e2e_test.cpp. That shape is
the one the entire eight-pass pipeline is tested against, and two details
of it look redundant and are not:

  * The activation is staged through `memref.alloc` + `memref.copy` rather
    than used straight from its global. cim-detect counts constant
    operands and requires exactly ONE (lib/Transforms/CIMDetect.cpp,
    `isWeightOperand`): a matmul of two constants has nothing to make
    resident, so it is not a CIM candidate at all. Using the global
    directly makes the whole module silently fail to offload -- no error,
    just no cim.program anywhere.
  * Weights are output-major [N x K], not [K x N]. That is `cim.mvm`'s
    convention, and what `cimrt_mvm` in runtime/src/simulator/simulator.cpp
    actually indexes: out[r] = sum_c W[r*cols + c] * act[c].

test/python/test_onnx_frontend.py pins the first property by comparing
this module's output against `build_module` byte for byte.
"""

import numpy as np

# MLIR symbol names are [A-Za-z_$.][A-Za-z0-9_$.]* -- ONNX tensor names are
# not, and routinely contain '/', ':' and '-'.
_SAFE_FIRST = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$.")
_SAFE_REST = _SAFE_FIRST | set("0123456789")


def sanitize_symbol(name, taken):
    """An MLIR-legal symbol for `name`, unique against everything in `taken`.

    `taken` is mutated. Uniqueness is the point, not prettiness: two
    distinct ONNX tensors that sanitize to the same string would end up
    sharing one `memref.global`, so two layers would silently compute
    against the same weights. The IR would verify and the numbers would be
    wrong -- exactly the failure mode this project refuses -- so a
    collision gets a numeric suffix rather than a warning.
    """
    # Every position is sanitized against the same (wider) alphabet, and a
    # leading character that is legal-but-not-legal-first is PREFIXED
    # rather than replaced. Replacing it would silently discard
    # information -- '0abc' and '9abc' would both become '_abc' and then
    # need the collision suffix to tell them apart, which is a worse name
    # for no reason.
    base = "".join(ch if ch in _SAFE_REST else "_" for ch in name) or "_"
    if base[0] not in _SAFE_FIRST:
        base = "_" + base

    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _dense_2d(arr):
    """A 2-D dense<> literal body, without the enclosing `dense<...>`."""
    rows = ", ".join(
        "[" + ", ".join(str(int(v)) for v in row) + "]" for row in arr)
    return f"[{rows}]"


def emit_module(weight, activation, weight_sym="w", act_sym="a", header=""):
    """A self-contained module computing `activation @ weight.T`.

    `weight` is [N, K] int8 -- ALREADY TRANSPOSED into cim.mvm's
    output-major convention. This function does not transpose: the ONNX
    -> output-major flip happens once, at a named boundary in
    onnx_import.py, so exactly one place in the front end can get it
    wrong and exactly one test has to guard it.

    `activation` is a length-K int8 vector.

    With the default symbol names and no header this returns text
    byte-identical to test/python/test_numerical_differential.py's
    `build_module()`.
    """
    weight = np.asarray(weight)
    activation = np.asarray(activation)
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [N, K], got shape {weight.shape}")
    if activation.ndim != 1:
        raise ValueError(
            f"activation must be 1-D [K], got shape {activation.shape}")

    n, k = weight.shape
    if activation.shape[0] != k:
        raise ValueError(
            f"activation length {activation.shape[0]} does not match the "
            f"weight's K of {k}")

    rows = _dense_2d(weight)
    acts = ", ".join(str(int(v)) for v in activation)
    return f"""\
{header}memref.global "private" constant @{weight_sym} : memref<{n}x{k}xi8> = dense<{rows}>
memref.global "private" constant @{act_sym} : memref<1x{k}xi8> = dense<[[{acts}]]>
func.func private @cim_print_i32(memref<*xi32>)
func.func @main() {{
  %w = memref.get_global @{weight_sym} : memref<{n}x{k}xi8>
  %aInit = memref.get_global @{act_sym} : memref<1x{k}xi8>
  %a = memref.alloc() : memref<1x{k}xi8>
  memref.copy %aInit, %a : memref<1x{k}xi8> to memref<1x{k}xi8>
  %out = memref.alloc() : memref<1x{n}xi32>
  linalg.matmul_transpose_b ins(%a, %w : memref<1x{k}xi8>, memref<{n}x{k}xi8>)
    outs(%out : memref<1x{n}xi32>)
  %u = memref.cast %out : memref<1x{n}xi32> to memref<*xi32>
  func.call @cim_print_i32(%u) : (memref<*xi32>) -> ()
  memref.dealloc %a : memref<1x{k}xi8>
  memref.dealloc %out : memref<1x{n}xi32>
  return
}}
"""
