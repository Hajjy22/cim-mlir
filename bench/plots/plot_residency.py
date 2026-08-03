#!/usr/bin/env python3
"""Plot weight-reprogramming cost by eviction policy from a cim-bench run.

Spec Sec. 10 requires every plot to be produced by a script in the repo, so
that a number in a talk can always be traced back to the command that made
it. Nothing here is drawn by hand.

Usage:
    cim-bench run --target erbium-8t --out results.json
    python3 bench/plots/plot_residency.py results.json -o residency.png

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


def check_validity(data):
    if not data.get("all_schedules_valid", False):
        print(
            "ERROR: this run contains a schedule that failed validation. "
            "A performance number without a correctness check is worthless; "
            "refusing to plot.",
            file=sys.stderr,
        )
        return False
    return True


def table(data):
    rows = data["results"]
    workloads = []
    for row in rows:
        if row["workload"] not in workloads:
            workloads.append(row["workload"])
    policies = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    by_key = {(r["workload"], r["policy"]): r for r in rows}

    header = f"{'workload':<16}" + "".join(f"{p:>12}" for p in policies)
    print(header)
    print("-" * len(header))
    for wl in workloads:
        line = f"{wl:<16}"
        for p in policies:
            entry = by_key.get((wl, p))
            line += f"{entry['programs'] if entry else '-':>12}"
        print(line)

    return workloads, policies, by_key


def plot(data, workloads, policies, by_key, out_path, provenance):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; printed the table only. "
            "Install it to render the plot.",
            file=sys.stderr,
        )
        return False

    width = 0.8 / max(len(policies), 1)
    positions = range(len(workloads))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, policy in enumerate(policies):
        values = [by_key[(wl, policy)]["programs"] for wl in workloads]
        offsets = [p + i * width for p in positions]
        ax.bar(offsets, values, width=width, label=policy)

    ax.set_xticks([p + 0.4 - width / 2 for p in positions])
    ax.set_xticklabels(workloads, rotation=15, ha="right")
    ax.set_ylabel("cim.program ops emitted (lower is better)")
    ax.set_title(
        f"Weight reprogramming cost by eviction policy\n"
        f"target: {data.get('target')} ({provenance} costs, "
        f"{data.get('inferences_modeled')} inferences) "
        f"commit {data.get('git_commit')}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="results.json produced by cim-bench")
    parser.add_argument("-o", "--out", default="residency.png", help="output image")
    args = parser.parse_args()

    data = load(args.results)
    provenance = check_provenance(data)
    if not check_validity(data):
        return 1

    workloads, policies, by_key = table(data)
    plot(data, workloads, policies, by_key, args.out, provenance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
