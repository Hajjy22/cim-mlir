# Target description format

The most reusable artifact in this project (spec Sec. 7). A vendor should
be able to support cim-mlir by writing one YAML file. See
`targets/erbium-8t.yaml` for the worked example referenced throughout this
page, and `include/cim/Target/TargetSpec.h` for the C++ struct this schema
is parsed into (`lib/Target/TargetYAMLParser.cpp`).

## Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Target identifier, e.g. `erbium-8t`. Matched against `cimrt_open(target_name, ...)` and `cim.device_open {target = ...}`. |
| `description` | string | Free-text description. |
| `version` | string | Target file schema/content version. |
| `class` | enum | One of `near_memory`, `digital_cim`, `analog_cim`, `dpu` — the compute-locality axis from `docs/abstraction.md`. |
| `provenance` | enum | One of `measured`, `simulated`, `estimated`. Every number in a target file must carry a source (spec Sec. 7 provenance discipline) — mark unmeasured targets `estimated` and say so loudly in every plot. |

## `tiles:`

| Field | Meaning |
|---|---|
| `count` | Number of physical tiles on the device. |
| `rows`, `cols` | Weight sub-matrix capacity per tile. |
| `weight_dtype`, `activation_dtype`, `accumulator_dtype` | Native precision (e.g. `i8`, `i8`, `i32`). **The functional simulator implements `i8` x `i8` -> `i32` only**, and `cimrt_open` now *refuses* a target declaring any other weight/activation width rather than reinterpreting it as `i8` and charging full cost — the silent-wrong-answer this schema previously allowed. The file still **parses**, so `cim-bench dump-target` can inspect hardware this build cannot run; parsing and executability are deliberately separate answers given at separate places. Widening the accepted set is real kernel work, and this refusal is what keeps that work visible instead of silently skipped. |
| `persistent` | Do weights survive across kernels / power-off? |
| `persistence` | `volatile` or `nonvolatile` — drives the program/mvm cost asymmetry (`docs/abstraction.md`). |

## `costs:`

| Field | Meaning |
|---|---|
| `program.latency_ns`, `program.energy_pj` | Cost of `cim.program` — the expensive, asymmetric op. |
| `mvm.latency_ns`, `mvm.energy_pj` | Cost of one `cim.mvm` against a resident tile. |
| `transfer.bandwidth_gbps`, `transfer.energy_pj_per_byte` | Per-byte host↔near transfer cost, charged on `cim.copy`. **`gbps` means gigaBYTES per second, not gigabits** — so transfer latency is exactly `bytes / bandwidth_gbps` ns (1 GB/s is 1 byte/ns). The field name is inherited from the spec and is misleading; the reading is pinned by `transfer_latency_pins_the_gigabytes_per_second_convention` in `test/unit/cimrt_test.cpp`, on a fixture with a non-unit bandwidth so the alternative readings actually differ. Note the static report counts only *explicit* `cim.copy` traffic — weight/activation staging is charged at runtime but has no `cim.copy` op to see, and the report says so via `transfer_bytes_excludes_implicit_staging`. |
| `requantize.latency_ns`, `requantize.energy_pj` | Cost of one `cim.requantize` — an ADC readout on an analog target, a narrowing/rounding step on a digital one (spec Sec. 5.3). Required like `program`/`mvm`: leaving it unset would silently cost every `cim.requantize` at zero rather than say so. |
| `reduce_partial.latency_ns`, `reduce_partial.energy_pj` | Cost of one chained add inside `cim.reduce_partial` — summing two already-computed partial accumulators (spec Sec. 5.4 rule 4), charged once per `cimrt_reduce_add` call an *N*-operand `cim.reduce_partial` lowers to (*N*-1 calls, not once per op site). Required like `requantize`, for the same reason. |
| `standby_leakage_uw_per_tile` | Idle power draw; `0.0` for non-volatile targets (the non-volatile advantage). |

## `precision:`

| Field | Meaning |
|---|---|
| `output_effective_bits` | 8 = lossless digital; `< 8` models ADC readout loss on analog targets. Drives `cim-legalize-precision`'s warning threshold. |

## `capabilities:`

| Field | Meaning |
|---|---|
| `double_buffer_program` | Can the target program tile *i+1* while computing on tile *i*? `cim-schedule` itself does not reorder or overlap anything in v0.1 (its own file header says so); `lib/Placement/CostReport.cpp` reads this to report a clearly-labeled PROJECTION of what overlapping would save (`steady_state_elapsed_ns_per_inference_if_overlapped` in `cim-bench`'s output), never a claim about what the compiled schedule does. |
| `partial_sum_in_place` | Can `cim.reduce_partial` accumulate without an extra buffer? Read by `cim-lower-to-target` (`lowerReducePartial`) and the interpreter alike: when true, an N-operand reduce allocates exactly ONE accumulator for the whole chain (`cimrt_reduce_add_inplace`, N-1 calls folded into it) instead of one fresh buffer per chained step (`cimrt_reduce_add`). |
| `autonomous_control` | Host-less execution supported (spec Sec. 3.1 control-model axis)? Parsed and echoed by `cim-bench dump-target`; not read by any pass -- v0.1's execution model has nothing host-less to drive. Pinned, not just claimed: `test/Transforms/cim-autonomous-control-is-unread.mlir` runs the full compiler chain against `test/targets/tiny-4x4.yaml` and `tiny-4x4-autonomous.yaml` (identical except this one flag) and diffs the two outputs byte for byte. |

## Units, and a discrepancy in the spec's worked example

All energy fields are **picojoules** (`energy_pj`) and all latency fields
are **nanoseconds** (`latency_ns`). `lib/Placement/CostReport.cpp` does its
arithmetic in those units throughout. Two fields are deliberately outside
that file's scope: `costs.requantize` and `costs.reduce_partial` are read
and charged by `cim-cost-report` (`lib/Transforms/CIMCostReport.cpp`),
which walks real compiled IR, not by `lib/Placement/CostReport.cpp`'s own
engine, which models weight-programming amortization only and has no
notion of a `cim.requantize` or a `cim.reduce_partial` at all — see that
file's own header comment.

Worth flagging before anyone builds a plot on it: the v0.1 spec's Section 17
worked example is internally inconsistent by 1000x on program energy. With
`erbium-8t.yaml`'s `program.energy_pj: 480000`:

- 480000 pJ = 480 nJ per program, and 8 programs = **3.84 µJ** of install energy.
- Section 17 states `× 480 nJ = 3.84 mJ`, which is 1000x larger than
  8 × 480 nJ.
- The rest of that table then follows from the mJ figure: amortized over 1M
  inferences it reports 3.84 nJ/inference and calls that 13% of the
  per-inference cost. Using the field as written (pJ), the same amortization
  gives 0.00384 nJ/inference — about 0.012%, i.e. negligible rather than
  material.
- The MVM numbers in the same table *are* self-consistent in pJ
  (`3100 pJ` = the 3.1 nJ the spec quotes), which points to the program-energy
  line being the error rather than the field's unit.

This repository implements the units as the field names declare them.
Whether the intended figure is 480 nJ or 480 µJ per program changes whether
install cost is a rounding error or a real 13% of the per-inference budget —
which is the crux of the amortization argument — so it is worth resolving
against real hardware before the number appears in a paper. Until then,
`provenance: estimated` on the Erbium file is doing exactly the job it
exists for.

## The supported YAML subset

`lib/Target/TargetYAMLParser.cpp` is a hand-rolled reader, not a YAML
implementation. Making the target file readable without an LLVM or libyaml
dependency is worth the tradeoff, but it means the supported grammar has to
be written down rather than inherited.

**Supported:** nested block mappings of scalars, `#` comments (at the start
of a line or after whitespace), blank lines, any consistent indentation
width, any key order, and single- or double-quoted scalars.

**Not supported, and rejected rather than guessed at:** sequences, flow
collections (`{a: 1}`, `[1, 2]`), anchors and aliases, multi-line scalars,
tabs for indentation, and documents that are not a single mapping.

**Scalar forms.** Numbers are plain decimal. The reader rejects:

| Form | Reason |
| --- | --- |
| `count: -1`, `count: +4` | `strtoul` negates a leading minus, so `-1` used to arrive as 4294967295 and pass the "must be greater than zero" check |
| `count: 4294967296` | silent truncation to 32 bits produces a plausible wrong number |
| `energy_pj: inf`, `nan` | accepted by `strtod`, and then every downstream sum is `inf`/`NaN` with no diagnostic |
| `persistent: yes` | only `true`/`false` are booleans here |

**Known divergences from PyYAML.** These are YAML 1.1 forms that a full
implementation resolves differently. They are outside the subset and are
pinned by `test/python/test_yaml_differential.py` so the list cannot grow
silently:

| Form | PyYAML (YAML 1.1) | This reader |
| --- | --- | --- |
| `012` | octal 10 | decimal 12 |
| `1_0` | 10 (digit separator) | rejected |
| `1e3` | the string `"1e3"` | the number 1000 |
| `yes` / `no` / `on` / `off` | booleans | rejected |

No shipped target file uses any of them, and a test enforces that.

## How the reader is verified

- `test/unit/parser_error_test.cpp` — a rejection table, one row per error
  branch, each asserting the diagnostic names the actual problem.
- `test/python/test_yaml_differential.py` — differential against PyYAML plus
  an independently written schema (`test/python/schema.py`). It checks the
  shipped files, hand-written mutations, and hypothesis-generated documents
  that vary field presence, key order, indentation width, comments and
  quoting. Both readers must agree on accept/reject *and* on every field
  value after schema coercion.

Run it with `pytest test/python` after a build (it drives
`cim-bench dump-target`, so a build directory must exist).

## Adding a new target

1. Copy `targets/erbium-8t.yaml` as a starting point.
2. Fill in every field with real numbers where you have them; mark
   anything you don't with `provenance: estimated` at minimum, and note it
   inline per-field if provenance varies within the file.
3. Validate it against the real reader — not just against a YAML parser,
   which says nothing about whether this repository can read it:
   `cim-bench dump-target --target-file targets/your-target.yaml`.
4. `cimrt_open("your-target", ...)` resolves to `targets/your-target.yaml`
   relative to the working directory in the v0.1 runtime (see the TODO in
   `runtime/src/simulator/simulator.cpp` about a real search path).
