"""Differential test: the C++ WorkloadJSON reader vs Python's own `json`.

Same shape as test_yaml_differential.py, one level down the stack: rather
than compare our target-YAML reader against PyYAML, this compares
`cim-bench analyze`'s hand-rolled JSON reader (lib/Placement/
WorkloadJSON.cpp, chosen instead of a library for the reasons in its own
header -- cim-bench and cimPlacement build with no LLVM/MLIR toolchain)
against the standard library's `json` module reading the exact same file.
`conftest.py`'s own module docstring explains why a subprocess test finds
different bugs than an in-process one: this compares the shipped artifact,
not our own understanding of it, against an implementation written by
other people for a different reason entirely.

What is compared is deliberately narrow -- layer count, skip count, and
the `model` string (including one with characters that stress both our
reader's `\\uXXXX` decoding and cim-bench's own JSON-writing escaper) --
because those are the only fields `cim-bench analyze` echoes back out in
its own JSON, and echoing them correctly end to end is the property that
actually matters to a caller.
"""

import json
import os

from conftest import run

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TARGET = os.path.join(REPO_ROOT, "test", "targets", "tiny-4x4.yaml")


def _analyze(cim_bench, workload_path, tmp_path):
    out = tmp_path / "out.json"
    result = run([cim_bench, "analyze", "--target-file", TARGET,
                 "--workload-file", str(workload_path), "--out", str(out)])
    return result, (json.load(open(out)) if result.returncode == 0 else None)


def _write(tmp_path, name, obj_or_text):
    path = tmp_path / name
    with open(path, "w") as handle:
        if isinstance(obj_or_text, str):
            handle.write(obj_or_text)
        else:
            json.dump(obj_or_text, handle)
    return path


def test_layer_and_skip_counts_agree_with_python_json(cim_bench, tmp_path):
    doc = {
        "model": "diff.onnx",
        "layers": [
            {"name": "conv1", "op_type": "QLinearConv", "k": 27, "n": 8},
            {"name": "conv2", "op_type": "QLinearConv", "k": 72, "n": 16},
            {"name": "mm1", "op_type": "MatMulInteger", "k": 4, "n": 2},
        ],
        "skipped": [
            {"name": "pool1", "op_type": "MaxPool", "reason": "no weights"},
        ],
    }
    path = _write(tmp_path, "w.json", doc)
    result, parsed = _analyze(cim_bench, path, tmp_path)
    assert result.returncode == 0, result.stderr

    expected = json.load(open(path))
    assert parsed["layers_analyzed"] == len(expected["layers"])
    assert parsed["layers_skipped"] == len(expected["skipped"])
    assert parsed["model"] == expected["model"]


def test_unicode_and_quote_bearing_names_round_trip(cim_bench, tmp_path):
    # \u escapes in the SOURCE file (what our C++ reader must decode) and
    # literal characters that must come back out correctly re-escaped in
    # cim-bench's own JSON writer -- exercising both ends of the pipe in
    # one file, the same way test_yaml_differential.py's own fixtures lean
    # on real, not synthetic-minimal, documents.
    text = (
        '{"model": "caf\\u00e9 \\"model\\"\\n.onnx", '
        '"layers": [{"name": "conv-1", "op_type": "QLinearConv", '
        '"k": 27, "n": 8}], '
        '"skipped": [{"name": "p\\u00f6\\u00f6l", "op_type": "MaxPool", '
        '"reason": "quote \\" and backslash \\\\ in reason"}]}'
    )
    path = _write(tmp_path, "w.json", text)

    # The independent oracle: Python's own json module on the SAME bytes.
    with open(path, encoding="utf-8") as handle:
        expected = json.load(handle)

    result, parsed = _analyze(cim_bench, path, tmp_path)
    assert result.returncode == 0, result.stderr
    assert parsed["model"] == expected["model"]
    assert parsed["layers_analyzed"] == 1
    assert parsed["layers_skipped"] == 1
    # Round-tripped through cim-bench's own JSON writer (jsonEscape) --
    # parse the OUTPUT with Python's json too, so a broken escaper (not
    # just a broken reader) would also fail this.
    assert parsed["skipped"][0]["name"] == expected["skipped"][0]["name"]
    assert parsed["skipped"][0]["reason"] == expected["skipped"][0]["reason"]


def test_malformed_json_is_refused_the_same_way_python_would_notice(
    cim_bench, tmp_path
):
    path = _write(tmp_path, "w.json", '{"model": "m", "layers": [')
    try:
        json.load(open(path))
        assert False, "the fixture itself must be invalid JSON"
    except json.JSONDecodeError:
        pass

    result, _ = _analyze(cim_bench, path, tmp_path)
    assert result.returncode != 0
    assert "workload" in result.stderr.lower() or "json" in result.stderr.lower() \
        or path.name in result.stderr


def test_missing_skipped_field_is_refused(cim_bench, tmp_path):
    # Valid, complete JSON by Python's own reading -- the refusal is a
    # SCHEMA decision (WorkloadDocument's own "no silent default" rule),
    # not a syntax one, so this is the case that would slip through if our
    # reader ever "helpfully" defaulted a missing field instead.
    doc = {"model": "m", "layers": []}
    path = _write(tmp_path, "w.json", doc)
    json.load(open(path))  # sanity: Python agrees this is well-formed JSON

    result, _ = _analyze(cim_bench, path, tmp_path)
    assert result.returncode != 0
    assert "skipped" in result.stderr
