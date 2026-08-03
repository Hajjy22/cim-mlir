# cim-mlir

**An open, retargetable compiler and runtime stack for compute-in-memory and processing-in-memory hardware.**

CIM/PIM hardware exists and some of it ships, but there is no open way to take a normal
neural network — a PyTorch model, an ONNX file — and run it on that hardware without
hand-writing assembly or using a vendor's closed, single-chip compiler. cim-mlir is a
three-part open-source project that fixes this:

1. **A compiler** — an MLIR dialect (`cim`) and lowering pipeline that takes standard ML
   graphs down to CIM/PIM primitives, with the hardware described by a portable target
   description file instead of hardcoded.
2. **A runtime** (`cimrt`) — a thin C API that executes the compiled artifact on real
   hardware or a simulator.
3. **A benchmark suite** (`cim-bench`) — reproducible workloads and a cost model so that
   "is CIM actually better here?" becomes a measurable question instead of a marketing claim.

It is not a general ML compiler (it plugs in below IREE/TVM/XLA as a backend), not a chip,
and not analog-first — the first working targets are digital/near-memory, because that is
what ships today. See [`docs/abstraction.md`](docs/abstraction.md) for the hardware
abstraction model, which is the actual intellectual contribution of this project — read it
before the code.

## Status

This repository is currently an **M0/M1 skeleton** (see [`docs/roadmap.md`](docs/roadmap.md)):
the `cim` dialect, its 8-pass lowering pipeline, the `cimrt` runtime API, and the target
description schema are all structurally defined, but pass bodies and most op verifiers are
`TODO` stubs. Nothing here compiles a real model yet.

## Quickstart

This project builds against LLVM/MLIR as an out-of-tree dialect (the same pattern used by
`mlir/examples/standalone`, CIRCT, and torch-mlir). It does **not** vendor or build LLVM —
you need your own LLVM/MLIR build (tested against LLVM 18.x) and must point CMake at it:

```sh
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cim-opt
cmake --build build --target check-cim-opt   # FileCheck test suite
```

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for a full from-source LLVM/MLIR
build if you don't already have one.

## Repository layout

```
include/cim/      ODS (TableGen) dialect + pass definitions
lib/               dialect, pass, and target-parsing implementations
runtime/           cimrt C API: functional simulator, cost model, hardware backends
tools/             cim-opt (the MLIR opt driver), cim-bench (benchmark harness)
test/              FileCheck tests for every dialect op
targets/           hardware target description YAML files
docs/              abstraction model, dialect reference, target format, roadmap
bench/             benchmark workloads and plots
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Chosen deliberately over GPL so that hardware
vendors can adopt this without a copyleft obstacle; that adoption is the entire point.
