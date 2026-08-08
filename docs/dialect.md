# `cim` dialect reference

This page is meant to be regenerated from ODS once the dialect stabilizes,
via the `add_mlir_doc(CIMOps CIMDialect Dialects/ -gen-dialect-doc)` target
declared in `include/cim/Dialect/CMakeLists.txt` (building the `mlir-doc`
CMake target emits `Dialects/CIMDialect.md` from the `.td` sources).

**Do not hand-edit op descriptions here** — edit the `.td` files instead:

- `include/cim/Dialect/CIMDialect.td` — dialect definition, base classes
- `include/cim/Dialect/CIMAttrs.td` — `#cim.space`, `#cim.persistence`
- `include/cim/Dialect/CIMTypes.td` — `!cim.tile`, `!cim.resident`, `!cim.device`
- `include/cim/Dialect/CIMOps.td` — all nine ops

Until the ODS-generated docs are wired into the build output, see
`docs/abstraction.md` for the hardware model these ops encode,
`test/Dialect/CIM/*.mlir` for one worked syntax example per op, and
[`docs/website/index.html`](website/index.html) for a hand-written
op-by-op reference (operands, result, one example each) alongside the
rest of the architecture — open it directly in a browser.
