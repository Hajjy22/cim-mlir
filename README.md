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

**Builds and passes its tests.** The `cim` dialect compiles against MLIR 18, parses and
round-trips all nine ops, and its verifiers reject malformed IR (10 FileCheck tests,
including negative cases for shape mismatch, accumulator saturation, and same-space copies).

**Partly implemented.** Two of the eight lowering passes are real: `cim-detect` finds
INT8 matmuls with a constant weight operand, and `cim-partition` lowers them into per-tile
`cim.program`/`cim.mvm` with partial-sum reduction and explicit memory-space transfers,
driven by the target file's tile geometry:

```sh
cim-opt model.mlir --cim-detect --cim-partition=target-yaml=targets/erbium-8t.yaml
```

The remaining six passes are registered but their bodies are `TODO` stubs, so nothing
compiles a real model end to end yet. `cim-partition`'s scope limits (matrix-vector,
output-major weights, exact tile multiples) are each refused with a warning rather than
silently mislowered — see [`docs/roadmap.md`](docs/roadmap.md).

## Quickstart

The core has no LLVM or MLIR dependency, so you can build and run the benchmark
immediately:

```sh
cmake -S . -B build -G Ninja -DCIM_ENABLE_MLIR=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/bin/cim-bench run --target erbium-8t --out results.json
```

That prints the reprogramming cost of each benchmark workload under each eviction policy.
On `erbium-8t`, `mm-fit` programs its 8 tiles once and then reprograms nothing across
1000 inferences, while under spill pressure optimal placement cuts weight programming by
47–64% against an LRU baseline — see [`bench/workloads/README.md`](bench/workloads/README.md)
for the table and its caveats.

To build the MLIR layer as well (dialect, lowering passes, `cim-opt`), you need MLIR 18.
This project does not vendor or build LLVM. On Debian/Ubuntu the distro packages are
enough — no from-source LLVM build required:

```sh
sudo apt-get install llvm-18-dev libmlir-18-dev mlir-18-tools
pip install lit    # for the FileCheck suite

cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
  -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT="$(which lit)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --build build --target check-cim-opt   # dialect FileCheck suite
```

If you have your own LLVM/MLIR build, point `MLIR_DIR` and `LLVM_DIR` at it instead.

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
