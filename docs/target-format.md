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
| `weight_dtype`, `activation_dtype`, `accumulator_dtype` | Native precision (e.g. `i8`, `i8`, `i32`). |
| `persistent` | Do weights survive across kernels / power-off? |
| `persistence` | `volatile` or `nonvolatile` — drives the program/mvm cost asymmetry (`docs/abstraction.md`). |

## `costs:`

| Field | Meaning |
|---|---|
| `program.latency_ns`, `program.energy_pj` | Cost of `cim.program` — the expensive, asymmetric op. |
| `mvm.latency_ns`, `mvm.energy_pj` | Cost of one `cim.mvm` against a resident tile. |
| `transfer.bandwidth_gbps`, `transfer.energy_pj_per_byte` | Per-byte host↔near transfer cost, charged on `cim.copy`. |
| `standby_leakage_uw_per_tile` | Idle power draw; `0.0` for non-volatile targets (the non-volatile advantage). |

## `precision:`

| Field | Meaning |
|---|---|
| `output_effective_bits` | 8 = lossless digital; `< 8` models ADC readout loss on analog targets. Drives `cim-legalize-precision`'s warning threshold. |

## `capabilities:`

| Field | Meaning |
|---|---|
| `double_buffer_program` | Can the target program tile *i+1* while computing on tile *i*? Used by `cim-schedule` to decide whether overlap is achievable. |
| `partial_sum_in_place` | Can `cim.reduce_partial` accumulate without an extra buffer? |
| `autonomous_control` | Host-less execution supported (spec Sec. 3.1 control-model axis)? |

## Units, and a discrepancy in the spec's worked example

All energy fields are **picojoules** (`energy_pj`) and all latency fields
are **nanoseconds** (`latency_ns`). `lib/Placement/CostReport.cpp` does its
arithmetic in those units throughout.

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

## Adding a new target

1. Copy `targets/erbium-8t.yaml` as a starting point.
2. Fill in every field with real numbers where you have them; mark
   anything you don't with `provenance: estimated` at minimum, and note it
   inline per-field if provenance varies within the file.
3. Validate it parses: `python3 -c "import yaml; yaml.safe_load(open('targets/your-target.yaml'))"`.
4. `cimrt_open("your-target", ...)` resolves to `targets/your-target.yaml`
   relative to the working directory in the v0.1 runtime (see the TODO in
   `runtime/src/simulator/simulator.cpp` about a real search path).
