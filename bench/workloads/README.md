# Benchmark workloads (spec Sec. 10)

The v0.1 workload table, planned but **not yet runnable** — each row is
blocked on the `cim-opt` pipeline (`cim-detect` through `cim-cost-report`)
being functional end-to-end (spec milestones M2/M3).

| Workload | Shape | What it probes |
|---|---|---|
| `mm-fit` | weights fit entirely in tiles | Best case: zero reprogramming after install |
| `mm-spill-2x` | weights 2× tile capacity | Reprogramming pressure, eviction policy quality |
| `mm-spill-8x` | weights 8× tile capacity | Thrashing behavior |
| `mlp-3layer` | 3 sequential matmuls | Cross-layer residency scheduling |
| `bert-ffn` | one transformer FFN block (2 matmuls) | A shape people actually care about |

## Harness requirements (once runnable)

- One command runs everything: `cim-bench run --target erbium-8t --out results.json`
  (CLI skeleton already exists: `tools/cim-bench/cim-bench.cpp`).
- Every result carries the target file hash, git commit, and date.
- Every plot script lives in `bench/plots/` — no plots produced by hand.
- Correctness check on every workload against a PyTorch reference, reported
  alongside performance. A performance number without a correctness check
  is worthless.
