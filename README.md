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

Early, and honest about which parts are real:

**Working and tested.** The placement engine — the scheduling problem this project exists
to solve (see [`docs/abstraction.md`](docs/abstraction.md)) — is implemented, unit-tested,
and runnable, along with the target-description reader, the analytical cost model, the
`cimrt` functional simulator's INT8 matrix-vector multiply, and the `cim-bench` harness.
None of it needs an LLVM toolchain.

**Scaffolded, not implemented.** The `cim` dialect and its types/ops/verifiers are defined
in ODS with FileCheck round-trip tests, and the 8-pass pipeline is declared — but the pass
bodies are `TODO` stubs, so nothing compiles a real model end to end yet. See
[`docs/roadmap.md`](docs/roadmap.md).

## Quickstart

The core has no LLVM or MLIR dependency, so you can build and run the benchmark
immediately:

```sh
cmake -S . -B build -G Ninja -DCIM_ENABLE_MLIR=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/tools/cim-bench/cim-bench run --target erbium-8t --out results.json
```

That prints the reprogramming cost of each benchmark workload under each eviction policy.
On `erbium-8t`, `mm-fit` programs its 8 tiles once and then reprograms nothing across
1000 inferences, while under spill pressure optimal placement cuts weight programming by
47–64% against an LRU baseline — see [`bench/workloads/README.md`](bench/workloads/README.md)
for the table and its caveats.

To build the MLIR layer as well (dialect, lowering passes, `cim-opt`), point CMake at your
own LLVM/MLIR build — this project does not vendor or build LLVM. Tested against LLVM 18.x:

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

The project splits in two: a core with no LLVM/MLIR dependency, and the MLIR layer on top
of it. The interesting algorithm lives in the core, which is why it can be tested without a
toolchain build.

```
include/cim/Placement/   placement engine + cost model API   (core, no MLIR)
include/cim/Target/      target description schema           (core, no MLIR)
include/cim/Dialect/     ODS (TableGen) dialect definitions  (MLIR)
include/cim/Transforms/  ODS pass declarations               (MLIR)
lib/Placement/           Belady/LRU/FIFO placement, cost model, workloads
lib/Target/              target file reader
lib/Dialect/, lib/Transforms/   dialect and lowering-pass implementations
runtime/                 cimrt C API: functional simulator, hardware backends
tools/cim-bench/         benchmark harness   (core, no MLIR)
tools/cim-opt/           the MLIR opt driver (MLIR)
test/unit/               unit tests for the core
test/Dialect/            FileCheck tests for every dialect op
targets/                 hardware target description YAML files
docs/                    abstraction model, dialect reference, target format, roadmap
bench/                   benchmark workloads and plot scripts
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Chosen deliberately over GPL so that hardware
vendors can adopt this without a copyleft obstacle; that adoption is the entire point.
