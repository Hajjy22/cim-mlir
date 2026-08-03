#!/usr/bin/env python3
"""Validate a cim-bench results.json.

CI previously ran cim-bench and checked only its exit code, which proves the
tool did not crash and nothing else. A results file with zero workloads, or
with a policy that lost to a weaker one, would have passed.

Every check here is an invariant that must hold for any correct run, not a
golden value -- pinning exact program counts would break on every legitimate
change to the workload shapes without catching more bugs.

Usage:
    python3 bench/check_results.py results.json [results2.json ...]
"""

import json
import sys

EXPECTED_WORKLOADS = {
    "mm-fit",
    "mm-spill-2x",
    "mm-spill-8x",
    "mlp-3layer",
    "bert-ffn",
}
EXPECTED_POLICIES = {"belady", "lru", "fifo"}


def check(path):
    """Returns a list of problems; empty means the file is sound."""
    problems = []

    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: could not be read as JSON: {exc}"]

    # Provenance: a result that cannot be traced back to the inputs that
    # produced it is not reproducible (spec Sec. 10).
    for field in ("target", "target_file_hash", "git_commit", "date",
                  "provenance", "inferences_modeled"):
        if not data.get(field):
            problems.append(f"{path}: missing or empty '{field}'")

    if data.get("all_schedules_valid") is not True:
        problems.append(
            f"{path}: all_schedules_valid is not true -- at least one schedule "
            "failed replay, so every number in this file is untrustworthy")

    results = data.get("results")
    if not results:
        problems.append(f"{path}: no results at all")
        return problems

    seen_workloads = {row.get("workload") for row in results}
    missing = EXPECTED_WORKLOADS - seen_workloads
    if missing:
        problems.append(f"{path}: missing workloads: {sorted(missing)}")

    by_key = {}
    for row in results:
        workload = row.get("workload")
        policy = row.get("policy")
        by_key[(workload, policy)] = row

        if row.get("schedule_valid") is not True:
            problems.append(
                f"{path}: {workload}/{policy} has schedule_valid != true")

        programs = row.get("programs")
        reuses = row.get("reuses")
        if programs is None or reuses is None:
            problems.append(f"{path}: {workload}/{policy} missing counts")
            continue
        if programs < 0 or reuses < 0:
            problems.append(f"{path}: {workload}/{policy} has negative counts")
        if row.get("install_programs", 0) > programs:
            problems.append(
                f"{path}: {workload}/{policy} installs more than it programs")

    # Every workload must be present under every policy, or the comparison
    # below is silently partial.
    for workload in sorted(seen_workloads):
        for policy in sorted(EXPECTED_POLICIES):
            if (workload, policy) not in by_key:
                problems.append(f"{path}: {workload} missing policy {policy}")

    # The central claim: optimal placement never loses to a runtime cache.
    for workload in sorted(seen_workloads):
        belady = by_key.get((workload, "belady"))
        if not belady:
            continue
        for baseline in ("lru", "fifo"):
            other = by_key.get((workload, baseline))
            if not other:
                continue
            if belady["programs"] > other["programs"]:
                problems.append(
                    f"{path}: {workload}: belady emitted "
                    f"{belady['programs']} programs, worse than {baseline}'s "
                    f"{other['programs']}")

    # mm-fit is sized to fit entirely in tiles, so after install it must
    # reprogram nothing. This is the headline result; if it regresses,
    # everything downstream of it is wrong.
    fit = by_key.get(("mm-fit", "belady"))
    if fit and fit.get("steady_state_programs_per_inference", -1) != 0:
        problems.append(
            f"{path}: mm-fit still reprograms in steady state "
            f"({fit.get('steady_state_programs_per_inference')}); it is sized "
            "to fit entirely in tiles")

    return problems


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    all_problems = []
    for path in argv[1:]:
        problems = check(path)
        all_problems.extend(problems)
        if not problems:
            print(f"OK: {path}")

    for problem in all_problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
