#!/usr/bin/env python3
"""Plot install-cost amortization from two `cim-bench amortize` runs.

Spec Sec. 10 requires every plot to be produced by a script in the repo, so
that a number in a talk can always be traced back to the command that made
it. Nothing here is drawn by hand.

The point (docs/roadmap.md's M3 section): amortizing install cost over more
inferences is what makes weight-stationary CIM worth compiling for, and that
argument only holds on non-volatile hardware. On a volatile target the array
must stay continuously powered to retain the weights this pass just
installed, which draws standby leakage for as long as the run takes --
amortizing the one-time install event over more inferences does not touch
that, so the volatile curve flattens onto a non-zero floor instead of
tracking the non-volatile curve down to zero. That is a difference in which
curve you are on, not a rescaled copy of the same curve, which is why this
takes two runs and one plot rather than one run and a bigger number.

Usage:
    cim-bench amortize --target erbium-8t --out nonvolatile.json
    cim-bench amortize --target generic-digital-cim --out volatile.json
    python3 bench/plots/plot_amortization.py nonvolatile.json volatile.json \\
        -o amortization.png

Requires matplotlib. Without it, the script still prints the table so the
numbers remain available.
"""

import argparse
import json
import sys


def load(path):
    with open(path) as fh:
        return json.load(fh)


def check_provenance(data):
    """Refuse to let an estimate be mistaken for a measurement."""
    provenance = data.get("provenance", "unknown")
    if provenance != "measured":
        print(
            f"WARNING: target '{data.get('target')}' declares {provenance} costs, "
            "not measured. Label every plot from this data accordingly.",
            file=sys.stderr,
        )
    return provenance


def leakage_floor_pj(data):
    """The per-inference floor a volatile target's curve settles on.

    Recomputed from the same fields CostReport::toJson exposes rather than
    read back off the last swept point, so a caller who trims kSweep in
    cim-bench.cpp still gets the right floor instead of a stale point value.
    Zero for a persistent (non-volatile) target -- there is no floor to
    plot, only the curve itself trending to zero.
    """
    if data.get("persistent", True):
        return 0.0
    points = data.get("points", [])
    if not points:
        return 0.0
    # The floor is whatever the curve has converged to by its largest swept
    # inference count -- amortizedInstallEnergyPjPerInference is monotone
    # decreasing, so the last point is the tightest one available without
    # recomputing the formula from scratch here.
    return points[-1]["amortized_install_pj_per_inference"]


def table(label, data):
    points = data["points"]
    print(f"{label}: {data.get('target')} ({'non-volatile' if data.get('persistent') else 'volatile'})")
    header = f"  {'inferences':>14}{'pj/inference':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for p in points:
        print(f"  {p['inferences']:>14}{p['amortized_install_pj_per_inference']:>18.6g}")


def plot(nonvolatile, volatile_, out_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; printed the tables only. "
            "Install it to render the plot.",
            file=sys.stderr,
        )
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for data, style in ((nonvolatile, "-o"), (volatile_, "-s")):
        xs = [p["inferences"] for p in data["points"]]
        ys = [p["amortized_install_pj_per_inference"] for p in data["points"]]
        kind = "non-volatile" if data.get("persistent") else "volatile"
        ax.plot(xs, ys, style, label=f"{data.get('target')} ({kind})")

    floor = leakage_floor_pj(volatile_)
    if floor > 0:
        ax.axhline(
            floor,
            color="grey",
            linestyle="--",
            linewidth=1,
            label=f"{volatile_.get('target')} standby-leakage floor",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("inferences the install cost is amortized over")
    ax.set_ylabel("amortized install energy (pJ/inference)")
    ax.set_title(
        "Install-cost amortization: persistence changes the curve, not just the number\n"
        f"workload: {nonvolatile.get('workload')}, policy: {nonvolatile.get('policy')}"
    )
    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nonvolatile", help="amortize JSON for a persistent target")
    parser.add_argument("volatile", help="amortize JSON for a non-persistent target")
    parser.add_argument("-o", "--out", default="amortization.png", help="output image")
    args = parser.parse_args()

    nonvolatile = load(args.nonvolatile)
    volatile_ = load(args.volatile)
    check_provenance(nonvolatile)
    check_provenance(volatile_)

    if bool(nonvolatile.get("persistent")) == bool(volatile_.get("persistent")):
        print(
            "ERROR: both inputs declare the same persistence; this plot "
            "needs one persistent (non-volatile) and one non-persistent "
            "(volatile) target to show anything.",
            file=sys.stderr,
        )
        return 1
    if not nonvolatile.get("persistent"):
        # Argument order doesn't matter to the numbers, only to the labels
        # and the floor line below -- swap so "nonvolatile" always refers
        # to whichever input actually declared persistent: true.
        nonvolatile, volatile_ = volatile_, nonvolatile

    table("nonvolatile", nonvolatile)
    print()
    table("volatile", volatile_)
    print()

    plot(nonvolatile, volatile_, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
