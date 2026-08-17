# Benchmark workloads

These run now, with no MLIR toolchain. They measure weight-residency pressure, which is
what the placement pass reasons about.

```sh
cmake -S . -B build -G Ninja -DCIM_ENABLE_MLIR=OFF && cmake --build build
./build/bin/cim-bench run --target erbium-8t --out results.json
python3 bench/plots/plot_residency.py results.json -o residency.png
```

The workloads are generated in `lib/Placement/Workloads.cpp` and sized against whatever
target file you pass, so the same table runs on an 8-tile near-memory device and a 64-tile
DPU without edits.

| Workload | Shape | What it probes |
|---|---|---|
| `mm-fit` | weights fit entirely in tiles | Best case: zero reprogramming after install |
| `mm-spill-2x` | weights 2× tile capacity | Eviction policy quality |
| `mm-spill-8x` | weights 8× tile capacity | Thrashing |
| `mlp-3layer` | 3 sequential matmuls | Cross-layer residency scheduling |
| `bert-ffn` | one transformer FFN block | A shape people actually care about |

These five are synthetic. For a *real* model's own layer shapes, use `cim-bench analyze`
with `cim-import-onnx --emit-workload` — same engine, same cost report, shapes from a real
network. See [`python/README.md`](../../python/README.md).

## Result on `erbium-8t` (8 tiles of 256×256, 1000 inferences)

`cim.program` ops emitted — lower is better:

| Workload | Belady | LRU | FIFO |
|---|---|---|---|
| `mm-fit` | **8** | 8 | 8 |
| `mm-spill-2x` | **8538** | 16000 | 16000 |
| `mm-spill-8x` | **56895** | 64000 | 64000 |
| `mlp-3layer` | **4370** | 12000 | 12000 |
| `bert-ffn` | **24776** | 32000 | 32000 |

Two things this shows, and together they are the argument for the project:

1. **`mm-fit` programs 8 times and then never again.** On a non-volatile target the
   weights are written once at install and survive power-off, so install cost is amortized
   over the life of the device rather than paid per inference.
2. **Compile-time knowledge beats a runtime cache under pressure.** LRU thrashes on a
   cyclic sweep — it evicts precisely the block it needs next, so every use misses. Belady
   can see the whole execution order, because the model graph is static.

## Simulator vs. the compiler

The table above is `cim-bench`'s own solve over a generated workload — not
`cim-placement` running on compiled MLIR. Those are different claims, so here is both:

| Workload | simulator | `cim-placement` on real IR | gap |
|---|---|---|---|
| `mm-fit` | 8 | 8 | 0% |
| `mm-spill-2x` | 8538 | 7 + 9×1000 = 9007 | +5.5% |
| `mm-spill-8x` | 56895 | 7 + 57×1000 = 57007 | +0.2% |
| `mlp-3layer` | 4370 | 7 + 5×1000 = 5007 | +14.6% |
| `bert-ffn` | 24776 | 7 + 25×1000 = 25007 | +0.9% |

The compiler emits `tiles-1` weights pinned plus `blocks-(tiles-1)` reprogrammed per
iteration — the proven optimum for any *single loop body*. The unrestricted solve wins the
remainder only by varying its per-iteration program count, which a fixed loop body cannot
express. `cim-cost-report` prints both numbers and the gap, so these rows are checkable
against the compiler rather than trusted.

So against LRU's 16000 on `mm-spill-2x`, the compiler's own output is a 43.7% reduction
where the simulator's is 47%.

## Caveats, stated plainly

- The `erbium-8t` cost numbers are **estimates, not measurements**
  (`provenance: estimated`). `cim-bench` warns on every run and the plot script repeats it.
- These are counts of weight *programming*, not wall-clock. They say nothing about
  arithmetic correctness — that is the functional simulator's job.
- Every schedule is replayed through `validatePlacement()` before its number is reported,
  and `cim-bench` exits non-zero if any fails.

## Persistence changes the optimum

The counts above do not depend on persistence — a `cim.program` costs the same either way.
What differs is how install cost amortizes:

```sh
cim-bench amortize --target erbium-8t --out nonvolatile.json
cim-bench amortize --target generic-digital-cim --out volatile.json
python3 bench/plots/plot_amortization.py nonvolatile.json volatile.json -o amortization.png
```

`erbium-8t` (non-volatile) amortizes `mm-fit`'s install cost down ~10× for every 10× more
inferences. `generic-digital-cim` (volatile, 4.5 µW/tile leakage) starts cheaper but its
curve flattens at a fixed per-inference floor: the array must stay powered to retain what
was programmed into it, and that cost scales with elapsed time, not with how many
inferences share the install. **The two curves cross** — that crossover, not a single
number, is what "persistence changes the optimum" means.
