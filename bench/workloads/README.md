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

## Still to come

- ONNX ingestion, so the shapes come from a real model file rather than
  being generated (spec M2).
- Numerical correctness against a PyTorch reference alongside these counts
  (spec Sec. 10) — needs the compiler pipeline to actually emit runnable
  artifacts.
- The volatile-vs-non-volatile comparison plot (spec M3): the same workload
  against `generic-digital-cim.yaml` to show that persistence changes the
  optimum, not just the magnitude.
