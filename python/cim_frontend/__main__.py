"""cim-import-onnx: an ONNX model in, the cim pipeline's MLIR out.

    PYTHONPATH=python python3 -m cim_frontend model.onnx --input act.npy \\
      | cim-opt - --cim-detect --cim-partition=target-yaml=<target> \\
      | cim-run --target-yaml=<target> -

Note which passes that pipeline does NOT include, because it matters: the
full eight-pass chain additionally runs cim-legalize-precision, which
inserts a cim.requantize clamping every accumulator to the target's
`precision.output_effective_bits` (8 in every shipped target). That is a
real modeled hardware effect, not a bug -- but it means the compiled
result stops matching an unquantized ONNX oracle as soon as any
accumulator leaves [-128, 127]. Compare against an oracle through
detect/partition/placement; add the rest when you want the modeled
precision loss.

A second, unrelated entry point, `--emit-workload`, skips MLIR entirely:
it does not compile the model, only reports the weight shapes cim-bench's
placement engine needs to estimate residency cost, and unlike the above
never refuses on an op it does not recognize -- see analyze.py's own
module docstring.
"""

import argparse
import json
import sys

import numpy as np

from .onnx_import import import_model
from .refusal import Refusal


def load_activation(path):
    """Activation values from a .npy or .json file, with its description."""
    if path.endswith(".npy"):
        return np.load(path), f"{path}"
    if path.endswith(".json"):
        with open(path) as handle:
            return np.asarray(json.load(handle)), f"{path}"
    raise Refusal(f"--input '{path}' must be a .npy or .json file.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cim-import-onnx",
        description="Import an ONNX MatMulInteger model as cim-pipeline MLIR.")
    parser.add_argument("model", help="path to a .onnx file")
    parser.add_argument(
        "--input", metavar="PATH",
        help="activation values (.npy or .json). Required: the emitted module "
             "bakes one inference, and defaulting the activation to zeros "
             "would make every downstream numerical check vacuous, since "
             "0 @ W == 0 for every W.")
    parser.add_argument(
        "--input-random", metavar="SEED", type=lambda s: int(s, 0),
        help="instead of --input, synthesize a pseudorandom int8 activation "
             "from SEED. Explicitly opt-in, and recorded in the emitted "
             "module's provenance header.")
    parser.add_argument("-o", "--output", metavar="PATH",
                        help="write here instead of stdout")
    parser.add_argument(
        "--emit-workload", action="store_true",
        help="skip MLIR emission; instead permissively walk the graph "
             "(cim_frontend.analyze) and write a JSON report of "
             "offloadable-layer weight shapes to -o/--output (or stdout), "
             "for `cim-bench analyze`. No --input/--input-random needed: "
             "unlike compiling, placement/cost analysis needs only weight "
             "shapes, never an activation. The report always names what "
             "was skipped alongside what was counted -- see "
             "cim_frontend/analyze.py's own docstring.")
    args = parser.parse_args(argv)

    try:
        import onnx
    except ImportError:
        print("cim-import-onnx: the 'onnx' package is required "
              "(pip install -r test/python/requirements-onnx.txt)",
              file=sys.stderr)
        return 2

    if args.emit_workload:
        try:
            with open(args.model, "rb") as handle:
                model_bytes = handle.read()
            model = onnx.load_from_string(model_bytes)

            from .analyze import analyze_model
            report = analyze_model(model, model_path=args.model)
            text = json.dumps(report, indent=2) + "\n"
        except Refusal as refusal:
            print(refusal.format(), file=sys.stderr)
            return 1

        if args.output:
            with open(args.output, "w") as handle:
                handle.write(text)
        else:
            sys.stdout.write(text)
        return 0

    try:
        with open(args.model, "rb") as handle:
            model_bytes = handle.read()
        model = onnx.load_from_string(model_bytes)

        if args.input:
            activation, source = load_activation(args.input)
        elif args.input_random is not None:
            # Mirror import_model's own dispatch: only the shape of the
            # FIRST activation is needed here, but which loader can name it
            # depends on whether the graph is a matmul chain, a single
            # matmul, or a single convolution.
            from .onnx_import import (ACCEPTED_CONV_OP, ACCEPTED_OP,
                                      load_matmul_chain, load_matmul_integer,
                                      load_qlinear_conv)
            matmul_count = sum(1 for n in model.graph.node
                              if n.op_type == ACCEPTED_OP)
            conv_count = sum(1 for n in model.graph.node
                            if n.op_type == ACCEPTED_CONV_OP)
            rng = np.random.default_rng(args.input_random)
            if matmul_count == 0 and conv_count >= 1:
                _, _, _, x_shape, _, _, _, _, _ = load_qlinear_conv(model)
                activation = rng.integers(
                    -128, 128, size=x_shape, dtype=np.int64).astype(np.int8)
            else:
                if matmul_count >= 2:
                    _, weights, _, _ = load_matmul_chain(model)
                    k0 = weights[0].shape[1]
                else:
                    _, weight, _ = load_matmul_integer(model)
                    k0 = weight.shape[1]
                activation = rng.integers(-128, 128, size=k0,
                                          dtype=np.int64).astype(np.int8)
            source = f"pseudorandom, seed={hex(args.input_random)}"
        else:
            raise Refusal(
                "no activation supplied. Pass --input <file>, or "
                "--input-random <seed> to synthesize one explicitly. There is "
                "deliberately no default: the emitted module bakes one "
                "inference, and a zero activation would make every numerical "
                "check downstream pass against any weights at all.")

        text = import_model(model, activation, model_path=args.model,
                            model_bytes=model_bytes, activation_source=source)
    except Refusal as refusal:
        print(refusal.format(), file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
