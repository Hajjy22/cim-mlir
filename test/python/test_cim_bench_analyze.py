"""`cim-bench analyze`: placement/cost over a REAL model's layer shapes.

Runs the actual shipped binary (not a reimplementation) against the
checked-in test/workloads/small-cnn-workload.json fixture -- the
`--emit-workload` output for onnx_fixtures.small_multi_layer_cnn_model(),
kept current by test_analyze.py's own regenerate-and-diff test -- so this
suite exercises the full Python-front-end-to-C++-placement-engine path
Phase 1 exists for, without a network fetch or an `onnx` dependency in the
C++-only jobs (the fixture is plain JSON).
"""

import json
import os

import pytest

from conftest import run

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO_ROOT, "test", "workloads",
                       "small-cnn-workload.json")
TARGET = os.path.join(REPO_ROOT, "test", "targets", "tiny-4x4.yaml")


def analyze(cim_bench, tmp_path, workload_file=FIXTURE, target_file=TARGET,
           extra_args=()):
    out = tmp_path / "out.json"
    cmd = [cim_bench, "analyze", "--target-file", target_file,
          "--workload-file", workload_file, "--out", str(out), *extra_args]
    result = run(cmd)
    data = json.load(open(out)) if result.returncode == 0 else None
    return result, data


def test_real_fixture_is_analyzed_end_to_end(cim_bench, tmp_path):
    result, data = analyze(cim_bench, tmp_path)
    assert result.returncode == 0, result.stderr

    assert data["model"] == "small-cnn.onnx"
    assert data["layers_analyzed"] == 3
    assert data["layers_skipped"] == 2
    assert len(data["skipped"]) == 2
    assert {s["op_type"] for s in data["skipped"]} == {"MaxPool"}
    assert "NOT end-to-end inference cost" in data["note"]
    assert data["all_schedules_valid"] is True

    # All three eviction policies ran and each produced a valid schedule
    # with real (positive) install/steady-state numbers -- the same shape
    # of result `run`'s built-in workloads produce, now for a workload
    # built from real per-layer shapes instead of a synthetic one.
    policies = {r["policy"] for r in data["results"]}
    assert policies == {"belady", "lru", "fifo"}
    for row in data["results"]:
        assert row["schedule_valid"] is True
        assert row["programs"] >= 1
        assert row["total_energy_pj"] > 0.0


def test_skip_disclosure_is_printed_to_stderr(cim_bench, tmp_path):
    result, _ = analyze(cim_bench, tmp_path)
    assert result.returncode == 0, result.stderr
    assert "NOT end-to-end inference cost" in result.stderr
    assert "pool1" in result.stderr and "MaxPool" in result.stderr
    assert "pool2" in result.stderr


def test_larger_k_on_one_layer_increases_the_block_count(cim_bench, tmp_path):
    """Mutation check for the K/N -> partitionBlockCount wiring: perturbing
    one real layer's K must move `programs` (more tile-sized blocks to
    place), not leave it unchanged -- the failure mode of a wiring bug that
    silently drops the shape and always builds some fixed-size workload
    instead."""
    with open(FIXTURE) as handle:
        doc = json.load(handle)
    doc["layers"][0]["k"] *= 8  # tiny-4x4.yaml's tiles are 4x4; this moves
                                 # conv1 from 1 tile-row to several.
    mutated = tmp_path / "mutated.json"
    with open(mutated, "w") as handle:
        json.dump(doc, handle)

    _, base = analyze(cim_bench, tmp_path, workload_file=FIXTURE)
    _, bigger = analyze(cim_bench, tmp_path, workload_file=mutated)

    base_programs = next(r["programs"] for r in base["results"]
                        if r["policy"] == "belady")
    bigger_programs = next(r["programs"] for r in bigger["results"]
                          if r["policy"] == "belady")
    assert bigger_programs > base_programs, (
        base_programs, bigger_programs)


def test_zero_offloadable_layers_is_refused(cim_bench, tmp_path):
    empty = tmp_path / "empty.json"
    with open(empty, "w") as handle:
        json.dump({"model": "empty.onnx", "layers": [],
                  "skipped": [{"name": "x", "op_type": "MaxPool",
                              "reason": "no weights"}]}, handle)
    result, _ = analyze(cim_bench, tmp_path, workload_file=str(empty))
    assert result.returncode != 0
    assert "0 offloadable layers" in result.stderr


def test_missing_workload_file_flag_is_rejected(cim_bench, tmp_path):
    out = tmp_path / "out.json"
    result = run([cim_bench, "analyze", "--target-file", TARGET, "--out",
                 str(out)])
    assert result.returncode != 0
    assert "--workload-file" in result.stderr


def test_nonexistent_workload_file_is_refused(cim_bench, tmp_path):
    result, _ = analyze(cim_bench, tmp_path,
                        workload_file="/nonexistent/w.json")
    assert result.returncode != 0
    assert "nonexistent" in result.stderr or "open" in result.stderr.lower()
