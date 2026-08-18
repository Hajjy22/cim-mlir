# Target description format

The most reusable artifact here: a vendor should be able to support cim-mlir by writing
one YAML file. Worked example: `targets/erbium-8t.yaml`. Parsed into
`include/cim/Target/TargetSpec.h` by `lib/Target/TargetYAMLParser.cpp`.

## Top level

| Field | Meaning |
|---|---|
| `name` | Target identifier, e.g. `erbium-8t`. Matched by `cimrt_open()` and `cim.device_open`. |
| `description` | Free text. |
| `version` | Schema/content version. |
| `class` | `near_memory`, `digital_cim`, `analog_cim`, or `dpu` — the compute-locality axis from [`abstraction.md`](abstraction.md). |
| `provenance` | `measured`, `simulated`, or `estimated`. Every number must carry a source. Mark unmeasured targets `estimated` and say so in every plot. |

## `tiles:`

| Field | Meaning |
|---|---|
| `count` | Number of physical tiles. |
| `rows`, `cols` | Weight sub-matrix capacity per tile. |
| `weight_dtype`, `activation_dtype`, `accumulator_dtype` | Native precision. The simulator implements `i8 × i8 -> i32` only, and `cimrt_open` **refuses** anything else rather than reinterpreting it. The file still *parses*, so `cim-bench dump-target` can inspect hardware this build cannot run. |
| `persistent` | Do weights survive across kernels / power-off? This is the field that drives the program-vs-mvm cost asymmetry. |
| `persistence` | `volatile` or `nonvolatile`. Documentation only — no pass reads it. The parser refuses a file where it disagrees with `persistent`. |

## `costs:`

All energies are **picojoules**, all latencies **nanoseconds**.

| Field | Meaning |
|---|---|
| `program.*` | Cost of one `cim.program` — the expensive, asymmetric op. |
| `mvm.*` | Cost of one `cim.mvm` against a resident tile. |
| `transfer.bandwidth_gbps` | **Gigabytes** per second, not gigabits — transfer latency is exactly `bytes / bandwidth_gbps` ns. The misleading name is inherited from the spec; the reading is pinned by a test. |
| `transfer.energy_pj_per_byte` | Charged per byte on `cim.copy`. The static report counts explicit `cim.copy` traffic only, and says so in its own output. |
| `requantize.*` | Cost of one `cim.requantize` — an ADC readout on analog, a narrowing step on digital. |
| `reduce_partial.*` | Cost of one chained add, charged once per `cimrt_reduce_add` call (an *N*-operand reduce makes *N*-1 calls). |
| `reduce_max.*` | Cost of one chained compare-and-select, same *N*-1 rule. A separate entry rather than reusing `reduce_partial`, because a comparator is a different datapath element from an adder — billing pooling to the adder would publish a number that reads as measured and is not. |
| `standby_leakage_uw_per_tile` | Idle power. `0.0` for non-volatile targets — that is the non-volatile advantage. |

Every cost entry is **required**. Leaving one unset would silently charge that op zero
rather than say so.

## `precision:`

| Field | Meaning |
|---|---|
| `output_effective_bits` | 8 = lossless digital; `< 8` models ADC readout loss. Drives `cim-legalize-precision`'s warning threshold. |

## `capabilities:`

| Field | Meaning |
|---|---|
| `double_buffer_program` | Can the target program tile *i+1* while computing on tile *i*? No pass reorders anything in v0.1; the cost report uses this only for a clearly-labeled *projection* of what overlap would save. |
| `partial_sum_in_place` | Can `cim.reduce_partial` accumulate without an extra buffer? When true, an *N*-operand reduce allocates one accumulator for the whole chain instead of one per step. |
| `max_in_place` | The identical question for `cim.reduce_max`: can this hardware fold a compare-and-select chain into its first operand's own storage? A **separate** flag from `partial_sum_in_place`, not the same capability reused — an adder and a comparator are different datapath elements (the same reason `reduce_max`'s cost entry is separate from `reduce_partial`'s above), so a target may support one fold and not the other. |
| `autonomous_control` | Host-less execution supported? Parsed and echoed, read by no pass — v0.1 has nothing host-less to drive. A test diffs two otherwise-identical targets to keep that true. |

## A known discrepancy in the spec

The v0.1 spec's Section 17 worked example is internally inconsistent by 1000× on program
energy: with `program.energy_pj: 480000`, 8 programs is 3.84 µJ, but the spec's table says
3.84 mJ. The MVM numbers in the same table *are* self-consistent in pJ, which points to
the program-energy line being the error.

This repository implements the units as the field names declare them. Whether the intended
figure is 480 nJ or 480 µJ decides whether install cost is a rounding error or 13% of the
per-inference budget — worth resolving against real hardware before it appears in a paper.

## The supported YAML subset

The reader is hand-rolled, not a YAML implementation — that is what keeps the target file
readable with no LLVM or libyaml dependency. So the grammar has to be written down.

**Supported:** nested block mappings of scalars, `#` comments, blank lines, any consistent
indentation width, any key order, single- or double-quoted scalars.

**Rejected rather than guessed at:** sequences, flow collections, anchors and aliases,
multi-line scalars, tabs for indentation, and anything that is not one mapping.

**Rejected scalars:**

| Form | Reason |
| --- | --- |
| `count: -1`, `count: +4` | `strtoul` negates a leading minus, so `-1` used to arrive as 4294967295 |
| `count: 4294967296` | silent 32-bit truncation gives a plausible wrong number |
| `energy_pj: inf`, `nan` | accepted by `strtod`, then every downstream sum is `inf`/`NaN` |
| `persistent: yes` | only `true`/`false` are booleans here |

**Known divergences from PyYAML** — YAML 1.1 forms outside this subset, pinned by
`test/python/test_yaml_differential.py` so the list cannot grow silently:

| Form | PyYAML | This reader |
| --- | --- | --- |
| `012` | octal 10 | decimal 12 |
| `1_0` | 10 | rejected |
| `1e3` | the string `"1e3"` | the number 1000 |
| `yes` / `no` / `on` / `off` | booleans | rejected |

No shipped target file uses any of them, and a test enforces that.

## Adding a target

1. Copy `targets/erbium-8t.yaml`.
2. Fill in real numbers where you have them; mark the rest `provenance: estimated`.
3. Validate against the real reader, not just against a YAML parser:
   `cim-bench dump-target --target-file targets/your-target.yaml`.
4. `cimrt_open("your-target", ...)` resolves the file relative to the working directory in
   the v0.1 runtime.

Verified by `test/unit/parser_error_test.cpp` (one row per error branch) and
`test/python/test_yaml_differential.py` (a differential against PyYAML plus an
independently written schema, over shipped files, mutations, and generated documents).
