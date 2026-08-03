# Roadmap

Milestones from the v0.1 spec (Section 13). Each has a public artifact —
nothing counts until it is public. **This commit's actual state is M0/M1**:
the repo/dialect skeleton exists and round-trips through `cim-opt`, but no
pass has real logic yet and nothing here compiles a real model.

## M0 — Environment and orientation (this commit, partial)
- [x] Repository scaffold matching this layout.
- [ ] LLVM/MLIR built from source, `cim-opt` actually compiling (blocked in
  the environment this scaffold was generated in — no MLIR install, no
  internet; see `.github/workflows/ci.yml` for the real build path).
- [ ] Toy tutorial chapters 1–7 completed.
- [ ] Erbium emulator running locally.

## M1 — Dialect skeleton (this commit, structurally complete)
- [x] `cim` dialect defined in ODS: types, attributes, all nine ops
  (`include/cim/Dialect/`).
- [x] Verifier hooks declared for the ops with real semantic rules
  (`cim.program`, `cim.mvm`, `cim.reduce_partial`) — bodies are `TODO`
  pending M2.
- [x] FileCheck round-trip test per op (`test/Dialect/CIM/`).
- [ ] CI green (workflow exists, unverified — see M0 note).

## M2 — Functional correctness (future)
- Functional simulator + `cimrt` runtime: partially scaffolded now
  (`runtime/src/simulator/simulator.cpp` has a real INT8 MVM reference
  implementation; the compiler side does not feed it real IR yet).
- Passes 1, 2, 5, 7 (`cim-detect`, `cim-partition`, `cim-insert-transfers`,
  `cim-lower-to-target`) need real logic — currently `TODO` stubs in
  `lib/Transforms/`.
- End-to-end: an ONNX INT8 matmul compiles and produces numerically correct
  output vs. PyTorch.

## M3 — The placement pass (future)
- `cim-placement` (Belady/MIN eviction) and `cim-cost-report` — both
  currently `TODO` stubs; `cim-placement` is the flagship pass (spec Sec. 3.3).
- `cim-bench` running the five v0.1 workloads (`bench/workloads/`).
- Headline result: reprogramming cost vs. model-size/tile-capacity ratio,
  optimal placement vs. naive baseline, on volatile vs. non-volatile targets.

## M4 — Second target and generalization (future)
- `targets/generic-digital-cim.yaml` already exists as a placeholder
  second-class target file; needs real (or better-estimated) numbers and a
  working lowering to prove retargetability.
- `cim-legalize-precision` with real `effective_bits` modeling.

## M5 — Community and real hardware (future)
- Real Erbium-8T hardware backend (`runtime/src/erbium/erbium_backend.cpp`
  currently stubs every entry point with `CIMRT_ERR_NO_DEVICE`).
- Upstream contribution to AiNEKKO/Erbium.
- Conference talk submission.

## M6 — Decision point (future)
Assess against the signals in spec Section 15 (external engagement, MLIR
learning curve, competitive landscape) and choose: raise/recruit, take a
role at a company in the space, or continue as an open-source maintainer.

## Naming note

The spec names the seventh pass `--cim-lower-to-<target>` (implying one
flag per target). MLIR pass registration requires a single fixed pass
name, so this is implemented as one `cim-lower-to-target` pass parameterized
by a `-target-yaml=<path>` option instead (`include/cim/Transforms/Passes.td`).
