"""An independent implementation of the target-file schema.

This is the oracle half of the YAML differential. It is written from
`docs/target-format.md` and spec Sec. 7, deliberately *not* transcribed from
`lib/Target/TargetYAMLParser.cpp` -- a port of the implementation under test
would reproduce its bugs and agree with it perfectly, which is the one way a
differential test can be worthless.

Division of labour with PyYAML: PyYAML does lexing and scalar resolution;
this module does schema validation and coercion. Both readers therefore
answer the same two questions -- does this document describe a device, and
what are its field values -- and the test compares the answers.

Scalars are validated as *text* rather than as resolved Python objects. The
schema says `tiles.count` is a non-negative 32-bit integer, and "is this
text a non-negative 32-bit integer" is a question with one answer, whereas
"did PyYAML happen to resolve this to a Python int" also says yes to
YAML 1.1 octal. Keeping the check textual means the only divergence the
test can see is a resolution difference, which is exactly what it is for.
"""

import os
import re

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TARGET_CLASSES = ("near_memory", "digital_cim", "analog_cim", "dpu")
PROVENANCES = ("measured", "simulated", "estimated")
PERSISTENCES = ("volatile", "nonvolatile")

UINT32_MAX = 4294967295

# Emission order. Must match tools/cim-bench's dump-target line for line;
# test_dump_target_emits_every_schema_field is what keeps them in step.
CANONICAL_ORDER = [
    "name",
    "class",
    "provenance",
    "tiles.count",
    "tiles.rows",
    "tiles.cols",
    "tiles.weight_dtype",
    "tiles.activation_dtype",
    "tiles.accumulator_dtype",
    "tiles.persistent",
    "tiles.persistence",
    "costs.program.latency_ns",
    "costs.program.energy_pj",
    "costs.mvm.latency_ns",
    "costs.mvm.energy_pj",
    "costs.requantize.latency_ns",
    "costs.requantize.energy_pj",
    "costs.transfer.bandwidth_gbps",
    "costs.transfer.energy_pj_per_byte",
    "costs.standby_leakage_uw_per_tile",
    "precision.output_effective_bits",
    "capabilities.double_buffer_program",
    "capabilities.partial_sum_in_place",
    "capabilities.autonomous_control",
]

UINT_FIELDS = ("tiles.count", "tiles.rows", "tiles.cols",
               "precision.output_effective_bits")
DOUBLE_FIELDS = ("costs.program.latency_ns", "costs.program.energy_pj",
                 "costs.mvm.latency_ns", "costs.mvm.energy_pj",
                 "costs.requantize.latency_ns", "costs.requantize.energy_pj",
                 "costs.transfer.bandwidth_gbps",
                 "costs.transfer.energy_pj_per_byte",
                 "costs.standby_leakage_uw_per_tile")
BOOL_FIELDS = ("tiles.persistent", "capabilities.double_buffer_program",
               "capabilities.partial_sum_in_place",
               "capabilities.autonomous_control")


def scalar_text(value):
    """The text a scalar would have had, or None if it is not a scalar.

    PyYAML has already resolved `8` to an int and `true` to a bool by the
    time we see it. Going back to text is what lets the checks below be
    textual; it is lossless for every form the documented subset allows.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return None  # list, dict, date, ...


def lookup(doc, path):
    """Walk a dotted path through nested mappings. Raises on a non-mapping."""
    node = doc
    parts = path.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, dict):
            raise ValueError(f"'{'.'.join(parts[:i])}' is not a mapping")
        if part not in node:
            return None
        node = node[part]
    return node


def parse_uint(path, text):
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"field '{path}': '{text}' is not a non-negative "
                         "decimal integer")
    value = int(text)
    if value > UINT32_MAX:
        raise ValueError(f"field '{path}': '{text}' does not fit in 32 bits")
    return value


def parse_double(path, text):
    # Python's float() rejects the C library's hex-float spelling and its
    # locale-dependent forms, which is what we want: the documented subset is
    # plain decimal.
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"field '{path}': '{text}' is not a number")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"field '{path}': '{text}' is not finite")
    return value


def parse_bool(path, text):
    if text not in ("true", "false"):
        raise ValueError(f"field '{path}': '{text}' is not a boolean")
    return text == "true"


def coerce(doc):
    """Validate a PyYAML document against the schema.

    Returns a flat {canonical_field: value} dict, or raises ValueError with
    the first problem found. Required fields are those `docs/target-format.md`
    marks required; everything else takes the documented default.
    """
    out = {}

    def text_at(path, required, default=None):
        raw = lookup(doc, path)
        if raw is None:
            if required:
                raise ValueError(f"missing required field '{path}'")
            return default
        text = scalar_text(raw)
        if text is None:
            raise ValueError(f"field '{path}' is not a scalar")
        return text

    out["name"] = text_at("name", required=True)

    class_text = text_at("class", required=True)
    if class_text not in TARGET_CLASSES:
        raise ValueError(f"field 'class': unknown target class "
                         f"'{class_text}'")
    out["class"] = class_text

    prov_text = text_at("provenance", required=False, default="estimated")
    if prov_text not in PROVENANCES:
        raise ValueError(f"field 'provenance': unknown provenance "
                         f"'{prov_text}'")
    out["provenance"] = prov_text

    for path in UINT_FIELDS:
        out[path] = parse_uint(path, text_at(path, required=True))

    for path in ("tiles.weight_dtype", "tiles.activation_dtype"):
        out[path] = text_at(path, required=True)
    out["tiles.accumulator_dtype"] = text_at("tiles.accumulator_dtype",
                                             required=False, default="")
    out["tiles.persistence"] = text_at("tiles.persistence", required=False,
                                       default="")

    for path in BOOL_FIELDS:
        text = text_at(path, required=False, default="false")
        out[path] = parse_bool(path, text)

    for path in DOUBLE_FIELDS:
        required = path != "costs.standby_leakage_uw_per_tile"
        text = text_at(path, required=required, default="0")
        out[path] = parse_double(path, text)

    # Semantic checks: a document that parses but describes an impossible
    # device must be rejected, or every cost number derived from it is
    # confidently wrong.
    if out["tiles.count"] == 0:
        raise ValueError("tiles.count must be greater than zero")
    if out["tiles.rows"] == 0 or out["tiles.cols"] == 0:
        raise ValueError("tiles.rows and tiles.cols must be greater than zero")
    if out["precision.output_effective_bits"] == 0:
        raise ValueError("precision.output_effective_bits must be greater "
                         "than zero")
    if out["tiles.persistence"] and out["tiles.persistence"] not in PERSISTENCES:
        raise ValueError("tiles.persistence must be 'volatile' or "
                         "'nonvolatile'")

    return out


def canonical_from_dict(flat):
    """Format a coerced spec exactly as `cim-bench dump-target` would."""
    lines = []
    for field in CANONICAL_ORDER:
        value = flat[field]
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif field in UINT_FIELDS:
            text = str(value)
        elif field in DOUBLE_FIELDS:
            # printf("%.10g") in C and "%.10g" % in Python agree bit for bit
            # on doubles, so formatting cannot be the source of a difference.
            text = "%.10g" % value
        else:
            text = str(value)
        lines.append(f"{field}={text}")
    return lines


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# Names that are safe to feed to both readers: lowercase identifiers that no
# YAML 1.1 resolver claims as a boolean, a null or a number. The lookahead is
# what keeps `no` and `on` out; without it the generator would manufacture
# divergences that are documented, expected, and not bugs.
SAFE_NAME = re.compile(
    r"(?!(?:yes|no|on|off|true|false|null|y|n)$)[a-z][a-z0-9_-]*")

_names = st.from_regex(SAFE_NAME, fullmatch=True).filter(
    lambda s: 1 <= len(s) <= 24 and not s.endswith(("-", "_")))

_dtypes = st.sampled_from(["i8", "u8", "i4", "i16", "i32"])

# Plain decimal only. Exponents, leading zeros, digit separators and the
# `.inf`/`.nan` spellings all mean something different to a YAML 1.1
# resolver than to strtod; see the divergence table in the test.
_uints = st.integers(min_value=1, max_value=100000).map(str)
_doubles = st.one_of(
    st.integers(min_value=0, max_value=10 ** 7).map(str),
    st.tuples(st.integers(min_value=0, max_value=10 ** 5),
              st.integers(min_value=0, max_value=999)).map(
                  lambda p: f"{p[0]}.{p[1]}"),
    st.tuples(st.integers(min_value=1, max_value=10 ** 4),
              st.integers(min_value=0, max_value=99)).map(
                  lambda p: f"-{p[0]}.{p[1]}"),
)
_bools = st.sampled_from(["true", "false"])


@st.composite
def _entries(draw, required, optional):
    """Build a mapping's entries: all required, a subset of optional, shuffled.

    Order is drawn rather than fixed because our reader keeps a nesting stack
    keyed on indentation, and a stack bug can easily be order-sensitive while
    passing on every hand-written file, which all happen to use one order.
    """
    entries = list(required)
    for key, strategy in optional:
        if draw(st.booleans()):
            entries.append((key, draw(strategy)))
    return draw(st.permutations(entries))


@st.composite
def _documents(draw):
    tiles = draw(_entries(
        required=[
            ("count", draw(_uints)),
            ("rows", draw(_uints)),
            ("cols", draw(_uints)),
            ("weight_dtype", draw(_dtypes)),
            ("activation_dtype", draw(_dtypes)),
        ],
        optional=[
            ("accumulator_dtype", _dtypes),
            ("persistent", _bools),
            ("persistence", st.sampled_from(list(PERSISTENCES))),
        ]))

    costs = draw(_entries(
        required=[
            ("program", [("latency_ns", draw(_doubles)),
                         ("energy_pj", draw(_doubles))]),
            ("mvm", [("latency_ns", draw(_doubles)),
                     ("energy_pj", draw(_doubles))]),
            ("requantize", [("latency_ns", draw(_doubles)),
                            ("energy_pj", draw(_doubles))]),
            ("transfer", [("bandwidth_gbps", draw(_doubles)),
                          ("energy_pj_per_byte", draw(_doubles))]),
        ],
        optional=[("standby_leakage_uw_per_tile", _doubles)]))

    capabilities = draw(_entries(
        required=[],
        optional=[("double_buffer_program", _bools),
                  ("partial_sum_in_place", _bools),
                  ("autonomous_control", _bools)]))

    top_required = [
        ("name", draw(_names)),
        ("class", draw(st.sampled_from(list(TARGET_CLASSES)))),
        ("tiles", tiles),
        ("costs", costs),
        ("precision", [("output_effective_bits", draw(_uints))]),
    ]
    top_optional = [
        ("description", st.lists(_names, min_size=1, max_size=4).map(" ".join)),
        ("version", st.sampled_from(["0.1", "1.0", "2"])),
        ("provenance", st.sampled_from(list(PROVENANCES))),
    ]
    # `capabilities:` with no children would be a bare key, which is a
    # nesting push to our reader and a null to PyYAML -- outside the subset,
    # so only emit the block when it has entries.
    if capabilities:
        top_required.append(("capabilities", capabilities))

    tree = draw(_entries(required=top_required, optional=top_optional))

    return {
        "tree": tree,
        "indent": draw(st.sampled_from([2, 4])),
        "comments": draw(st.booleans()),
        "blank_lines": draw(st.booleans()),
        "inline_comments": draw(st.booleans()),
        "quote": draw(st.booleans()),
    }


def _needs_quoting_is_safe(text):
    """Quoting is only round-trip safe for text with no quotes or hashes."""
    return not any(c in text for c in "\"'#:\n")


def emit_yaml(document):
    """Render a generated document as text both readers must accept."""
    lines = []

    def emit(entries, depth):
        pad = " " * (document["indent"] * depth)
        for key, value in entries:
            if document["blank_lines"]:
                lines.append("")
            if document["comments"]:
                lines.append(f"{pad}# {key}")
            if isinstance(value, list):
                lines.append(f"{pad}{key}:")
                emit(value, depth + 1)
                continue
            text = value
            if document["quote"] and _needs_quoting_is_safe(text):
                text = f'"{text}"'
            suffix = "  # note" if document["inline_comments"] else ""
            lines.append(f"{pad}{key}: {text}{suffix}")

    emit(document["tree"], 0)
    return "\n".join(lines) + "\n"


# Each example spawns a subprocess, so the default 100 examples costs real
# wall time. 40 locally, more in CI where a nightly can afford it.
MAX_EXAMPLES = int(os.environ.get("CIM_HYPOTHESIS_EXAMPLES", "40"))


def target_documents():
    """@given over the documented subset, with subprocess-friendly settings."""
    def decorate(fn):
        return settings(
            deadline=None,  # a subprocess round trip blows any deadline
            max_examples=MAX_EXAMPLES,
            suppress_health_check=[HealthCheck.too_slow],
        )(given(document=_documents())(fn))
    return decorate
