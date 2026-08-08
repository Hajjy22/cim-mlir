# Benchmark workloads (spec Sec. 10)

The v0.1 workload table. These are **runnable now** — they measure weight
residency pressure, which is what the placement pass reasons about, and need
no MLIR toolchain:

```sh
cmake -S . -B build -G Ninja -DCIM_ENABLE_MLIR=OFF && cmake --build build
./build/tools/cim-bench/cim-bench run --target erbium-8t --out results.json
python3 bench/plots/plot_residency.py results.json -o residency.png
```

The workloads are generated in `lib/Placement/Workloads.cpp` and sized
against whichever target file is passed, so the same table runs on an 8-tile
near-memory device and a 64-tile DPU without editing anything.

| Workload | Shape | What it probes |
|---|---|---|
| `mm-fit` | weights fit entirely in tiles | Best case: zero reprogramming after install |
| `mm-spill-2x` | weights 2× tile capacity | Reprogramming pressure, eviction policy quality |
| `mm-spill-8x` | weights 8× tile capacity | Thrashing behavior |
| `mlp-3layer` | 3 sequential matmuls | Cross-layer residency scheduling |
| `bert-ffn` | one transformer FFN block (2 matmuls) | A shape people actually care about |

## Result on `erbium-8t` (8 tiles of 256×256, 1000 inferences)

`cim.program` ops emitted — lower is better:

| Workload | Belady | LRU | FIFO |
|---|---|---|---|
| `mm-fit` | **8** | 8 | 8 |
| `mm-spill-2x` | **8538** | 16000 | 16000 |
| `mm-spill-8x` | **56895** | 64000 | 64000 |
| `mlp-3layer` | **4370** | 12000 | 12000 |
| `bert-ffn` | **24776** | 32000 | 32000 |

Two things this shows, and they are the argument for the project:

1. **`mm-fit` programs 8 times and then never again.** All 1000 inferences
   after installation reprogram nothing. On a non-volatile target that means
   the weights are written once at install and survive power-off — the
   install cost is amortized over the life of the device, not paid per
   inference.
2. **Compile-time knowledge beats a runtime cache under pressure.** LRU
   thrashes completely on a cyclic sweep — it evicts precisely the block it
   is about to need next, so every single use misses. Belady, which can see
   the whole execution order because the model graph is static, cuts
   programs by 47% on `mm-spill-2x` and 64% on `mlp-3layer`.

Caveats, stated plainly: the `erbium-8t` cost numbers are **estimates, not
measurements** (`provenance: estimated`), so the derived energy figures
inherit that uncertainty — `cim-bench` prints a warning to that effect on
every run and the plot script repeats it. These counts are of weight
*programming*, not end-to-end wall-clock; they say nothing about arithmetic
correctness, which is the functional simulator's job
(`runtime/src/simulator`). Every schedule above is replayed through
`validatePlacement()` before its number is reported, and `cim-bench` exits
non-zero if any schedule fails.

One more line to draw, because it is easy to read this table as "what the
compiler does" when it is actually "what the standalone simulator computes":
these numbers come from `cim-bench`'s own Belady/LRU/FIFO solve over a
generated workload, not from running `cim-placement` on compiled MLIR.
`cim-placement` on real compiled IR does reproduce the `mm-fit` result above
— a model whose weights entirely fit in tiles reprograms once, however many
inferences run, checked by executing hoisted IR through a real interpreter
(`test/mlir/pipeline_e2e_test.cpp`).

On the spill rows it comes close rather than matching, and the distance is
measured, not hand-waved. `cim-placement` emits `tiles-1` weights hoisted
plus `blocks-(tiles-1)` reprogrammed per iteration:

| Workload | simulator (this table) | `cim-placement` on compiled IR | gap |
|---|---|---|---|
| `mm-fit` | 8 | 8 | 0% |
| `mm-spill-2x` | 8538 | 7 + 9×1000 = 9007 | +5.5% |
| `mm-spill-8x` | 56895 | 7 + 57×1000 = 57007 | +0.2% |
| `mlp-3layer` | 4370 | 7 + 5×1000 = 5007 | +14.6% |
| `bert-ffn` | 24776 | 7 + 25×1000 = 25007 | +0.9% |

The `tiles-1` / `blocks-(tiles-1)` split is verified directly against the
compiler at two very different sizes — 16 blocks and 64 blocks over 8
tiles, both giving exactly 7 hoisted (see
`test/Transforms/cim-placement-spill-loop.mlir`). The `mlp-3layer` and
`bert-ffn` rows apply the same formula to their own block counts rather
than being measured one by one.

So against LRU's 16000 on `mm-spill-2x`, the compiler's own output is a
43.7% reduction where this table's 8538 is 47%. That is the honest version
of the claim, and it is a much smaller caveat than the previous wording
here implied.

`blocks-(tiles-1)` per iteration is the proven optimum for any single loop
body — a weight no op in the body programs holds a tile permanently, and
at most `tiles-1` weights can. The unrestricted solve wins the remainder
only by varying its per-iteration program count, which a fixed loop body
cannot express. `cim-placement` now runs the flattened solve on the real
IR and `cim-cost-report` prints both numbers plus the gap, so these rows
can be checked against the compiler rather than trusted.
`docs/roadmap.md`'s M3 section has the full explanation.

## Volatile-vs-non-volatile: persistence changes the optimum

The counts above are Belady/LRU/FIFO's reprogramming counts, which do not
depend on persistence — a `cim.program` costs the same whether the tile
that holds it is volatile SRAM or non-volatile MRAM/RRAM. What differs is
whether holding a weight resident *between* uses is free, and that shows up
not in the counts but in how install cost amortizes:

```sh
cim-bench amortize --target erbium-8t --out nonvolatile.json
cim-bench amortize --target generic-digital-cim --out volatile.json
python3 bench/plots/plot_amortization.py nonvolatile.json volatile.json -o amortization.png
```

`erbium-8t` (non-volatile, `standby_leakage_uw_per_tile: 0.0`) amortizes
`mm-fit`'s one-time install cost down by ~10x for every 10x more inferences
it is spread over — the number that makes weight-stationary CIM worth
compiling for. `generic-digital-cim` (volatile, `4.5` µW/tile) amortizes the
same way at first, since its install cost is cheaper to begin with, but its
curve flattens and never drops below a fixed per-inference floor: the array
must stay continuously powered to retain what was just programmed into it,
and that leakage cost scales with elapsed time, not with how many
inferences share the install event. The two curves cross — at few
inferences the volatile target wins on install cost, at many the
non-volatile target wins because its curve keeps falling while the volatile
one is stuck at its floor. That crossover, not just a different number at
one fixed inference count, is what "persistence changes the optimum" means
(`docs/roadmap.md`'s M3 section has the full derivation and the unit tests
that pin it).

## Still to come

- ONNX ingestion, so the shapes come from a real model file rather than
  being generated (spec M2).
- Numerical correctness against a PyTorch reference alongside these counts
  (spec Sec. 10) — needs the compiler pipeline to actually emit runnable
  artifacts.
