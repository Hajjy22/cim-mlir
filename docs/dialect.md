# `cim` dialect reference

Ten ops:

| Op | What it does |
|---|---|
| `cim.device_open` | Opens a device described by a target YAML file |
| `cim.tile_alloc` | Reserves a physical tile |
| `cim.tile_free` | Releases it |
| `cim.program` | Writes a weight sub-matrix into a tile — the expensive, asymmetric op |
| `cim.mvm` | Matrix-vector multiply against a resident tile |
| `cim.reduce_partial` | Sums partial accumulators across the contraction dimension |
| `cim.reduce_max` | Signed elementwise max — a MaxPool kernel window |
| `cim.requantize` | Rounds and clamps an accumulator down to the output precision |
| `cim.copy` | An explicit transfer between memory spaces |
| `cim.barrier` | Waits for outstanding device work |

Plus the attributes `#cim.space<host|near|insitu>` and `#cim.persistence`, and the types
`!cim.tile`, `!cim.resident`, `!cim.device`.

**Do not hand-edit op descriptions here** — edit the ODS sources instead:

- `include/cim/Dialect/CIMDialect.td` — dialect definition, base classes
- `include/cim/Dialect/CIMAttrs.td` — attributes
- `include/cim/Dialect/CIMTypes.td` — types
- `include/cim/Dialect/CIMOps.td` — all ten ops

This page will be regenerated from ODS once the dialect stabilizes, via the
`add_mlir_doc(...)` target already declared in `include/cim/Dialect/CMakeLists.txt`.

Until then: [`abstraction.md`](abstraction.md) for the hardware model these ops encode,
`test/Dialect/CIM/*.mlir` for one worked syntax example per op, and
[`website/index.html`](website/index.html) for a full op-by-op reference with operands,
results, and examples.
