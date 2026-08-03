"""Differential test: our target reader vs PyYAML + an independent schema.

CI used to have a `lint-targets` job that ran `yaml.safe_load` over
`targets/*.yaml` and printed OK. That proves the files are valid YAML and
says nothing whatsoever about the reader that actually consumes them --
`lib/Target/TargetYAMLParser.cpp` is a hand-rolled reader for a strict
subset, and every bug it can have (a mis-scoped nesting stack, a comment
stripped from inside a value, a scalar coerced with the wrong function) is
invisible to that job. This replaces it.

The oracle is PyYAML plus `schema.py`'s reimplementation of the schema in
`docs/target-format.md`. Two independent implementations of one written
contract disagree exactly where at least one of them is wrong -- which is
the entire value of a differential -- so the interesting work is making the
comparison fair:

  * Compare after schema coercion, never raw YAML scalars. `version: 0.1`
    is a float to PyYAML and the string "0.1" to us; that difference is
    meaningless because the schema says version is a string.
  * Both sides must agree on ACCEPT/REJECT, not just on values. A reader
    that accepts a document describing zero tiles is broken even if every
    field it did read matches.
  * Generated documents are constrained to the subset `docs/target-format.md`
    documents. PyYAML implements YAML 1.1, where `yes` is a boolean, `012`
    is octal and `1e3` is a string; our reader implements none of that and
    is not trying to. Those divergences are pinned deliberately in
    `test_known_divergences_stay_known` rather than being generated into
    every run as noise.

Three layers, weakest to strongest: the shipped files, hand-written
mutations, and generated documents.
"""

import glob
import os
import re

import pytest
import yaml

from conftest import REPO_ROOT, run
from schema import (
    SAFE_NAME,
    canonical_from_dict,
    coerce,
    emit_yaml,
    target_documents,
)


def dump_target(cim_bench, path):
    """Run the real reader. Returns (ok, canonical_lines_or_stderr)."""
    result = run([cim_bench, "dump-target", "--target-file", path])
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, result.stdout.strip().splitlines()


def oracle(text):
    """Read with PyYAML + the independent schema. Same return shape."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return False, f"pyyaml: {exc}"
    if not isinstance(doc, dict):
        return False, "pyyaml: document is not a mapping"
    try:
        return True, canonical_from_dict(coerce(doc))
    except ValueError as exc:
        return False, f"schema: {exc}"


def compare(cim_bench, tmp_path, text, label):
    """Assert both readers agree on this document. Returns their verdict."""
    path = os.path.join(str(tmp_path), "target.yaml")
    with open(path, "w") as fh:
        fh.write(text)

    ours_ok, ours = dump_target(cim_bench, path)
    theirs_ok, theirs = oracle(text)

    if ours_ok != theirs_ok:
        verdict = "accepted" if ours_ok else "rejected"
        other = "rejected" if ours_ok else "accepted"
        pytest.fail(
            f"{label}: our reader {verdict} a document PyYAML+schema "
            f"{other}.\n--- document ---\n{text}\n"
            f"--- ours ---\n{ours}\n--- oracle ---\n{theirs}")

    if ours_ok and ours != theirs:
        diff = "\n".join(
            f"  ours={o!r} oracle={t!r}"
            for o, t in zip(ours, theirs) if o != t)
        pytest.fail(
            f"{label}: both readers accepted but disagree on values.\n"
            f"--- document ---\n{text}\n--- differing fields ---\n{diff}")

    return ours_ok


# ---------------------------------------------------------------------------
# Layer 1: the files we actually ship
# ---------------------------------------------------------------------------

SHIPPED = sorted(glob.glob(os.path.join(REPO_ROOT, "targets", "*.yaml")) +
                 glob.glob(os.path.join(REPO_ROOT, "test", "targets", "*.yaml")))


def test_there_are_shipped_targets_to_check():
    # Without this, a bad glob would turn every parametrized test below into
    # a silent no-op and the suite would stay green while testing nothing.
    assert len(SHIPPED) >= 4, f"expected the shipped targets, found {SHIPPED}"


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: os.path.basename(p))
def test_shipped_target_matches_pyyaml(cim_bench, tmp_path, path):
    with open(path) as fh:
        text = fh.read()
    assert compare(cim_bench, tmp_path, text, os.path.basename(path)), (
        f"{path} was rejected by both readers -- a shipped target must load")


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: os.path.basename(p))
def test_shipped_target_stays_in_the_documented_subset(path):
    """The shipped files must not rely on any YAML 1.1 exotica.

    This is what keeps `test_known_divergences_stay_known` honest: those
    divergences are only acceptable while no file we ship depends on the
    behaviour they cover.
    """
    with open(path) as fh:
        lines = fh.readlines()

    for lineno, raw in enumerate(lines, 1):
        line = raw.split("#")[0].rstrip() if not raw.lstrip().startswith("#") \
            else ""
        if not line.strip():
            continue
        assert "\t" not in line, f"{path}:{lineno}: tab indentation"
        assert not line.lstrip().startswith("-"), f"{path}:{lineno}: sequence"
        assert ":" in line, f"{path}:{lineno}: not a key: value pair"
        value = line.split(":", 1)[1].strip()
        if not value:
            continue
        assert value.lower() not in ("yes", "no", "on", "off", "null", "~"), (
            f"{path}:{lineno}: {value!r} means different things to the two "
            "readers")
        if re.fullmatch(r"[0-9_]+", value):
            assert not value.startswith("0") or value == "0", (
                f"{path}:{lineno}: {value!r} is octal to PyYAML, decimal to us")


# ---------------------------------------------------------------------------
# Layer 2: mutations of a known-good document
# ---------------------------------------------------------------------------

def base_document():
    with open(os.path.join(REPO_ROOT, "targets", "erbium-8t.yaml")) as fh:
        return fh.read()


def mutate(old, new):
    text = base_document()
    assert old in text, f"mutation anchor {old!r} is no longer in the file"
    return text.replace(old, new, 1)


MUTATIONS = [
    # Both readers must reject these, and for a shared reason.
    ("drop the name", lambda: mutate("name: erbium-8t\n", "")),
    ("drop the class", lambda: mutate("class: near_memory", "")),
    ("drop tiles.count", lambda: mutate("  count: 8\n", "")),
    ("drop precision", lambda: mutate("  output_effective_bits: 8", "")),
    ("unknown class", lambda: mutate("class: near_memory",
                                     "class: warp_drive")),
    ("unknown provenance", lambda: mutate("provenance: estimated",
                                          "provenance: vibes")),
    ("zero tiles", lambda: mutate("  count: 8", "  count: 0")),
    ("zero rows", lambda: mutate("  rows: 256", "  rows: 0")),
    ("negative tile count", lambda: mutate("  count: 8", "  count: -1")),
    ("tile count past 32 bits",
     lambda: mutate("  count: 8", "  count: 4294967296")),
    ("infinite program energy",
     lambda: mutate("energy_pj: 480000", "energy_pj: inf")),
    ("non-numeric latency",
     lambda: mutate("latency_ns: 12000", "latency_ns: quick")),
    ("non-integer tile count", lambda: mutate("  count: 8", "  count: eight")),
    ("bad persistence",
     lambda: mutate("  persistence: nonvolatile", "  persistence: sometimes")),
    ("zero effective bits",
     lambda: mutate("  output_effective_bits: 8", "  output_effective_bits: 0")),
    ("tab indentation", lambda: mutate("  count: 8", "\tcount: 8")),
    # Valid YAML that our subset does not cover; the schema layer rejects it
    # because a list is not a scalar, so the two still agree.
    ("a sequence where a scalar belongs",
     lambda: mutate("  count: 8", "  count:\n    - 8\n    - 9")),
    # These stay valid: both readers must still accept, and still agree.
    ("comments and blank lines everywhere",
     lambda: "# leading\n\n" + base_document().replace(
         "costs:", "# about the costs\n\ncosts:") + "\n# trailing\n"),
    ("quoted scalars",
     lambda: mutate("name: erbium-8t", 'name: "erbium-8t"')),
    ("single-quoted scalars",
     lambda: mutate("class: near_memory", "class: 'near_memory'")),
    ("trailing whitespace", lambda: mutate("  count: 8", "  count: 8   ")),
    ("deeper but consistent indentation",
     lambda: base_document().replace("\n  ", "\n    ").replace(
         "\n    program:", "\n  program:")),
]


@pytest.mark.parametrize("label,build", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_mutations_are_judged_identically(cim_bench, tmp_path, label, build):
    compare(cim_bench, tmp_path, build(), label)


def test_the_base_document_is_accepted(cim_bench, tmp_path):
    # The control for the table above: if erbium-8t itself failed to load,
    # every "both rejected" row would pass for the wrong reason.
    assert compare(cim_bench, tmp_path, base_document(), "base")


# ---------------------------------------------------------------------------
# Layer 3: generated documents
# ---------------------------------------------------------------------------

@target_documents()
def test_generated_documents_are_read_identically(cim_bench, tmp_path_factory,
                                                  document):
    """Hypothesis over the documented subset.

    The strategy varies field values, field presence, key order within a
    mapping, indentation width, comment placement and blank lines -- every
    axis a hand-rolled line-oriented reader can get wrong -- while staying
    inside the scalar forms both implementations agree on.
    """
    tmp_path = tmp_path_factory.mktemp("gen")
    compare(cim_bench, tmp_path, emit_yaml(document), "generated")


# ---------------------------------------------------------------------------
# Known divergences: pinned, not discovered
# ---------------------------------------------------------------------------

DIVERGENCES = [
    # (label, value for tiles.count, what our reader does)
    ("octal-looking integer", "012",
     "decimal 12; PyYAML reads YAML 1.1 octal 10"),
    ("underscored integer", "1_0",
     "rejected; PyYAML reads YAML 1.1 digit separators as 10"),
]

BOOL_DIVERGENCES = [
    ("yes", "rejected; PyYAML reads a YAML 1.1 boolean true"),
    ("no", "rejected; PyYAML reads a YAML 1.1 boolean false"),
    ("on", "rejected; PyYAML reads a YAML 1.1 boolean true"),
]


@pytest.mark.parametrize("label,value,behaviour", DIVERGENCES,
                         ids=[d[0] for d in DIVERGENCES])
def test_known_integer_divergences_stay_known(cim_bench, tmp_path, label,
                                              value, behaviour):
    """Pin the places the two readers legitimately differ.

    These are YAML 1.1 scalar forms that `docs/target-format.md` puts
    outside the supported subset. They are pinned rather than fixed because
    matching PyYAML here would mean implementing YAML 1.1 resolution in a
    reader whose entire purpose is to avoid a YAML dependency. What must not
    happen is the set growing silently: a new divergence appearing in a form
    the shipped files use would be a real bug, and
    `test_shipped_target_stays_in_the_documented_subset` is what stops that.
    """
    text = mutate("  count: 8", f"  count: {value}")
    path = os.path.join(str(tmp_path), "target.yaml")
    with open(path, "w") as fh:
        fh.write(text)

    ours_ok, ours = dump_target(cim_bench, path)
    theirs_ok, theirs = oracle(text)

    assert ours_ok != theirs_ok or ours != theirs, (
        f"{label} ({behaviour}) no longer diverges -- if the reader was "
        "fixed deliberately, delete this row; the divergence table must "
        "describe reality")


@pytest.mark.parametrize("value,behaviour", BOOL_DIVERGENCES,
                         ids=[d[0] for d in BOOL_DIVERGENCES])
def test_known_boolean_divergences_stay_known(cim_bench, tmp_path, value,
                                              behaviour):
    text = mutate("  persistent: true", f"  persistent: {value}")
    path = os.path.join(str(tmp_path), "target.yaml")
    with open(path, "w") as fh:
        fh.write(text)

    ours_ok, _ = dump_target(cim_bench, path)
    assert not ours_ok, (
        f"our reader now accepts {value!r} as a boolean ({behaviour}); "
        "if that was deliberate, update docs/target-format.md and this row")


# ---------------------------------------------------------------------------
# The reader is not a rubber stamp
# ---------------------------------------------------------------------------

def test_dump_target_emits_every_schema_field(cim_bench):
    """A field the dumper forgets is a field this whole file cannot check."""
    path = os.path.join(REPO_ROOT, "targets", "erbium-8t.yaml")
    ok, lines = dump_target(cim_bench, path)
    assert ok, lines

    emitted = {line.split("=", 1)[0] for line in lines}
    expected = {line.split("=", 1)[0] for line in
                canonical_from_dict(coerce(yaml.safe_load(open(path).read())))}
    assert emitted == expected, (
        "dump-target and the oracle disagree about which fields exist: "
        f"only in dump-target {sorted(emitted - expected)}, "
        f"only in the schema {sorted(expected - emitted)}")


def test_a_name_that_looks_like_a_number_stays_a_string(cim_bench, tmp_path):
    # `name: 0.1` is a float to PyYAML and a string to us -- the schema says
    # string, so coercion must resolve it and both must report the same text.
    compare(cim_bench, tmp_path, mutate("name: erbium-8t", "name: 2024"),
            "numeric name")


def test_safe_name_pattern_matches_what_the_generator_promises():
    # Guards the generator's own contract: if SAFE_NAME let through `yes` or
    # a bare number, layer 3 would report divergences that are not bugs.
    for bad in ("yes", "no", "null", "012", "1.5", "~", "a#b", "a:b"):
        assert not SAFE_NAME.fullmatch(bad), f"{bad!r} must not be generated"
    for good in ("erbium-8t", "tiny_4x4", "a", "x1"):
        assert SAFE_NAME.fullmatch(good), f"{good!r} should be generatable"
