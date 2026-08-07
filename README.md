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

**Partly implemented.** Seven of the eight lowering passes are real: `cim-detect` finds
INT8 matmuls with a constant weight operand, `cim-partition` lowers them into per-tile
`cim.program`/`cim.mvm` with partial-sum reduction and explicit memory-space transfers,
and `cim-placement` — the pass the project exists for — rewrites that IR from a Belady
schedule, erasing weight programming it has proven redundant and assigning tile ids from
the solution:

```sh
cim-opt model.mlir --cim-detect \
  --cim-partition=target-yaml=targets/erbium-8t.yaml \
  --cim-placement=target-yaml=targets/erbium-8t.yaml
```

On two matmuls sharing a weight matrix that fits in the device's tiles, that turns four
`cim.program` ops into two, and the runtime's `programs` counter drops with it. Under
spill pressure it still wins — on a 4-block model over 2 tiles it saves 2 of 8 programs
where an LRU cache would save none. Every case is executed both with and without the pass
and the outputs must be **identical**: placement is an optimization and is not allowed to
change an answer.

Reuse is found across matmuls within a block, and now across iterations of an `scf.for`
too: a `cim.program` is hoisted above the loop when its tile is provably untouched by
anything else for the whole iteration and the loop's trip count is a compile-time
constant known positive. That reproduces the headline claim — a model that fits entirely
reprograms once, no matter how many inferences the loop runs — on compiled IR, checked by
executing the loop through a real interpreter, not just by inspecting its shape. It does
**not** replicate a full N-inference Belady solve, which is what the spill-workload
figures below still come from (`docs/roadmap.md`'s M3 section draws the exact line).

`cim-cost-report` (Pass 8) also rewrites nothing but walks the final, already-placed IR
and emits the project's publishable numbers as JSON — the same `CostReport`/`toJson`
format `cim-bench` already prints, not a second cost model that could drift from it.
Every `cim.program`/`cim.mvm` is weighted by the product of its enclosing loops'
compile-time-constant trip counts before being counted, because a `cim.program` hoisted
above an `scf.for` executes once regardless of where it sits textually while one left
inside executes trip-count times — a plain walk-and-count would misreport exactly the IR
`cim-placement`'s loop hoisting now produces. Checked against `cim-run --profile`'s real
`cimrt` counters for straight-line, spill and loop-hoisted modules alike
(`test/mlir/cost_report_e2e_test.cpp`): predicted and executed `programs`/`mvms` must
agree exactly.

`cim-schedule` (Pass 4) keeps source order — v0.1 never reorders or overlaps anything,
that is v0.2's double-buffering work — and inserts `cim.barrier` conservatively: a device's
outstanding `cim.program`/`cim.mvm` work is tracked as an open, un-synchronized run until
something actually depends on its result (a direct operand, or a use nested inside an
`scf.for` body — the loop op itself never lists a value only its body references, which is
exactly the case that needs a barrier *before* the loop, not just inside it), at which
point a barrier is inserted immediately before the dependent op. `cim.barrier` is a no-op
in the functional simulator, so a numerical differential cannot tell a correctly-placed
barrier from a missing one — that is what the structural tests
(`test/Transforms/cim-schedule.mlir`) are for, and a mutation test confirms they actually
catch a broken scheduler that a numerical-only check would silently let through
(`test/mlir/schedule_e2e_test.cpp` covers the "does not change the answer" half instead).

`cim-legalize-precision` (Pass 6) inserts `cim.requantize` after every "terminal"
accumulator — a `cim.reduce_partial` result, or a `cim.mvm` result no `cim.reduce_partial`
consumes (the single-K-tile case) — using `scale=1.0`/`zero_point=0` always, since v0.1 has
no per-layer calibration step anywhere in the pipeline to derive anything else from; what
this legalizes is the op shape spec Sec. 6 calls for, and the clamp that
`output_effective_bits` genuinely does encode. On a target declaring fewer than 8 effective
bits it emits a warning naming exactly how many of the 256 i8 encodings are unreachable
through that readout path, rather than silently degrading. The interpreter executes
`cim.requantize` for real — round-half-away-from-zero, then clamp to the signed range
`effective_bits` can hold — checked against an independent reference implementation over
both a fractional scale/nonzero zero_point and the clamp itself in both directions
(`test/mlir/legalize_precision_e2e_test.cpp`), since unlike `cim.barrier` this op genuinely
changes values and there is no "does not change the answer" invariant to fall back on.
Not yet wired into the live pipeline: `cim-partition` emits its own `cim.copy` back to a
host buffer typed to match the original i32 accumulator, with no knowledge that this pass
might requantize that value down to i8 first, so it is tested on hand-written IR shaped the
way `cim-partition`'s output locally is (`test/Transforms/cim-legalize-precision.mlir`) —
the same kind of documented v0.1 integration gap `cim-insert-transfers` has too.

`cim-insert-transfers` (Pass 5) inserts a `cim.copy` wherever a `cim.mvm`'s activation
operand is not already `#cim.space<near>` (spec Sec. 3.4) and rewires the `mvm` to read the
copy's result, then hoists that copy above an enclosing `scf.for` when the source is
loop-invariant with respect to it — inserted once before the loop rather than once per
iteration, and shared by every `mvm` inside that loop reading the exact same invariant
source rather than duplicated per site. On today's real pipeline this pass has nothing to
do: `cim-partition` already stages every activation into near space itself before this pass
would ever see it, so — like `cim-legalize-precision` above — it is tested on hand-written
IR that manufactures the space mismatch `cim-partition`'s current output never produces
(`test/Transforms/cim-insert-transfers.mlir`). `cim.copy` is a faithful byte-for-byte move,
not a value-changing operation, so the numerical side of verification is an invariance
check like `cim-schedule`'s: `test/mlir/insert_transfers_e2e_test.cpp` confirms running a
module with and without the pass computes identical numbers, while
`test/Transforms/cim-insert-transfers.mlir`'s structural checks are what actually catch a
broken hoist — a mutation reverting the loop-invariance check produces correct numbers with
redundant copies inside the loop, which the numerical suite alone would silently accept.

The remaining pass (`cim-lower-to-target`) is registered but its body is a `TODO` stub, so
nothing compiles a real model end to end yet. `cim-partition`'s scope limits
(matrix-vector, output-major weights, exact tile multiples) are each refused with a warning
rather than silently mislowered — see [`docs/roadmap.md`](docs/roadmap.md).

`cimrt` (the C ABI everything above eventually calls through) went through a hardening
pass: a use-after-free when a buffer outlived its device, an allocation failure that
threw a C++ exception across the `extern "C"` boundary instead of returning
`CIMRT_ERR_OOM`, misleading status codes, and a profiling window that never actually
windowed anything are all fixed, each with a regression test that failed against the
unfixed code first (`test/unit/cimrt_test.cpp`). The interpreter also gained a guard
against sub-byte element types (`i1`/`i4`), which used to silently compute zero-byte
allocations via `bitWidth / 8` truncating to zero.

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

## Verification

A compiler that produces plausible-looking IR is worth nothing; the question is
always whether the numbers are right. Six suites answer different parts of that,
and CI runs all of them.

| Suite | What it would catch that nothing else does |
| --- | --- |
| `ctest` — `cim-unit-tests` | Placement, cost model, target reader and `cimrt` error branches. Includes a property test asserting Belady placement equals exhaustive-search optimal over ~4700 instances, and a mutation test of the schedule validator. |
| `ctest` — `cim-mlir-tests` | Runs the real pass pipeline and then *executes* the emitted IR through the interpreter and `cimrt`, comparing against a C++ reference. Shapes matching is not arithmetic matching. |
| `check-cim-opt` (lit/FileCheck) | Dialect round-tripping, verifier rejections, pass output structure, and `cim-run`'s refusals. |
| `pytest test/python` | Differentials against oracles written by other people: the target reader vs PyYAML, and the compiled pipeline vs numpy. |
| ASan + UBSan, valgrind | Memory and undefined-behaviour bugs that produce correct answers today. |
| gcovr, clang-tidy, cppcheck | Untested branches and defects no test was written for. |

```sh
# Everything, on a build configured as above
ctest --test-dir build --output-on-failure
cmake --build build --target check-cim-opt
pip install -r test/python/requirements.txt && pytest test/python

# Instrumented builds (all default OFF; a release build is unaffected)
cmake -S . -B build-asan -G Ninja -DCIM_SANITIZER=address,undefined \
  -DCIM_ENABLE_WERROR=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake -S . -B build-cov  -G Ninja -DCIM_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
```

Coverage is gated at 85% line / 78% branch over `lib/Placement`, `lib/Target`,
`lib/Interpreter` and `runtime/src` (currently 90.3% / 83.3%). `lib/Dialect` is
mostly TableGen output and `lib/Transforms` is still one-eighth empty stubs,
so both are reported and not gated — a threshold there would measure how many
stubs exist rather than how well anything is tested.

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
lib/Interpreter/         executes the emitted IR against cimrt (MLIR)
runtime/                 cimrt C API: functional simulator, hardware backends
tools/cim-bench/         benchmark harness   (core, no MLIR)
tools/cim-opt/           the MLIR opt driver (MLIR)
tools/cim-run/           runs a compiled module and prints its results (MLIR)
test/unit/               unit tests for the core
test/mlir/               numerical end-to-end test: compile, execute, compare
test/Dialect/            FileCheck tests for every dialect op
test/Transforms/         FileCheck tests for the lowering passes
test/Run/                FileCheck tests for cim-run
test/python/             differentials against PyYAML and numpy
targets/                 hardware target description YAML files
docs/                    abstraction model, dialect reference, target format, roadmap
bench/                   benchmark workloads and plot scripts
```

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Chosen deliberately over GPL so that hardware
vendors can adopt this without a copyleft obstacle; that adoption is the entire point.
