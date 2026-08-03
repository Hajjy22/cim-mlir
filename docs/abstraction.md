# The cim-mlir hardware abstraction model

This is the most valuable file in this repository — more valuable than the
code, because it is the part that is hard to reproduce. Everything else in
this project follows from the model described here. Get this right and the
project has value even if the code around it is mediocre.

## The problem in one paragraph

CIM/PIM hardware exists and some of it ships. What does not exist is a way
to take a normal neural network — a PyTorch model, an ONNX file — and run
it on that hardware without hand-writing assembly or using a vendor's
closed, single-chip compiler. Every vendor rebuilds this layer privately and
badly. Academic work (PrIM, SimplePIM, DaPPA) solved it for exactly one chip
(UPMEM). The result is that CIM/PIM hardware has no software gravity, and
hardware without software gravity does not get adopted.

## Five axes of variation

Real CIM/PIM hardware varies along five axes. A useful abstraction must
parameterize all five and assume none.

| Axis | Range in real hardware | Why the compiler must care |
|---|---|---|
| **Compute locality** | In the bitcell array (analog crossbar) → at the sense amps / bitline (digital in-memory) → beside the bank (near-memory logic) → on the DIMM (UPMEM DPU) | Determines what a single "operation" costs and what data must move |
| **Weight residency** | Weight-stationary and persistent → reloaded per layer | The dominant scheduling constraint. Reprogramming cost varies by ~1000× |
| **Persistence** | Volatile SRAM → non-volatile MRAM/RRAM/flash | Non-volatility means weights survive power-off: enables intermittent/always-on execution models that have no equivalent in a normal compiler |
| **Precision** | Analog 3–8 bit effective → digital INT8 / FP8 / BF16 | Determines whether quantization/legalization passes are needed and whether accuracy must be modeled |
| **Control model** | Host-driven offload → autonomous in-memory cores (host-less) | Determines whether the compiler emits a kernel or a whole program |

## The unifying abstraction: the Tile

All CIM/PIM hardware is modeled as an array of **tiles**. A tile is:

> A fixed-capacity 2D compute-and-storage unit that can hold a weight
> sub-matrix and perform a matrix-vector multiply (MVM) against it.

This is deliberately the *lowest common denominator*. A UPMEM DPU is a tile
(with an unusual, wide, programmable MVM). A 256×256 digital SRAM CIM macro
is a tile. An analog crossbar is a tile. A near-memory RISC-V + iRAM cluster
is a tile. See `!cim.tile` in `include/cim/Dialect/CIMTypes.td` and the
`tiles:` block in `targets/*.yaml` for how this shows up in the dialect and
target schema respectively.

A tile carries: `rows`/`cols` (weight sub-matrix capacity), `weight_dtype`/
`activation_dtype` (precision), `persistent`/`persistence` (does it survive
power-off?), and cost fields for programming vs. computing (see below).

## The critical asymmetry the compiler exists to exploit

On a GPU, loading weights and computing cost roughly comparable amounts.
**On CIM hardware they differ by orders of magnitude**, and on non-volatile
CIM the asymmetry is extreme: writes to MRAM/RRAM are slow and
energy-hungry, but once written the weights sit there at zero standby
leakage indefinitely.

This means the central optimization problem is not "schedule the math." It is:

> **Given N physical tiles and a model with M weight matrices where M > N,
> decide which weights live where and when, to minimize total reprogramming
> cost.**

This is a scheduling + placement problem that no mainstream ML compiler
solves, because no mainstream hardware has this cost structure. **This is
the thesis of the entire project.** The `cim-placement` pass
(`lib/Transforms/CIMPlacement.cpp`) is the pass that implements it, and if
this project builds nothing else of value, it builds that pass.

## Memory spaces

Three address spaces are modeled as MLIR memory-space attributes on
`memref` (`#cim.space<host|near|insitu>` in `include/cim/Dialect/CIMAttrs.td`):

- `#cim.space<host>` — normal host DRAM
- `#cim.space<near>` — memory local to the compute-in-memory unit (DPU
  scratchpad, tile buffer)
- `#cim.space<insitu>` — the weight array itself, where compute happens

Transfers between spaces are explicit ops (`cim.copy`). Nothing implicit.
This is the single design decision that makes cost modeling honest: every
byte that moves between spaces is visible in the IR, and therefore every
byte that moves is accounted for in the cost report (`cim-cost-report`).

## What this project deliberately does not model (v0.1)

Out of scope for the v0.1 contract (see `docs/roadmap.md`): training,
floating point, convolutions, attention/softmax, multi-chip execution,
dynamic shapes, analog noise modeling, and autotuning. Analog CIM is
modeled as a target *class* (`class: analog_cim` in the target schema) but
the first working targets are digital/near-memory, because that is what
ships. Cycle-accurate hardware timing is explicitly out of scope in favor
of an analytical cost model (spec Sec. 9.2) — if real timing is later
needed, integrate Ramulator2 rather than rebuilding a simulator.
