//===- Passes.td - CIM lowering pipeline ------------------*- tablegen -*-===//
//
// The 8-pass lowering pipeline from spec Section 6, run in this order:
//   cim-detect -> cim-partition -> cim-placement -> cim-schedule ->
//   cim-insert-transfers -> cim-legalize-precision -> cim-lower-to-target ->
//   cim-cost-report
//
// Note: the spec names the second-to-last pass `--cim-lower-to-<target>`
// (one flag per target). MLIR pass registration requires one fixed pass
// name, so this is implemented as a single `cim-lower-to-target` pass
// parameterized by a `target-yaml` option instead — see docs/roadmap.md.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_TRANSFORMS_PASSES
#define CIM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CIMDetect : Pass<"cim-detect", "ModuleOp"> {
  let summary = "Pattern-match linalg.matmul / batch_matmul (later: conv_2d) "
                "with a constant INT8 weight operand and annotate as CIM candidates";
  let description = [{
    Spec Sec. 6, Pass 1. Detection and conversion are deliberately separate
    passes so each is independently testable: this pass only annotates
    `{cim.candidate = true}`, it does not convert to `cim` dialect ops yet.
  }];
  let constructor = "mlir::cim::createCIMDetectPass()";
}

def CIMPartition : Pass<"cim-partition", "ModuleOp"> {
  let summary = "Tile a logical weight matrix into tile-sized sub-matrices "
                "and emit cim.program/cim.mvm/cim.reduce_partial";
  let description = [{
    Spec Sec. 6, Pass 2. tiles_k = ceil(K/tile_rows), tiles_n = ceil(N/tile_cols).
    Ragged edges are zero-padded rather than special-cased (correct and
    simple beats fast and a bug farm, per spec).
  }];
  let constructor = "mlir::cim::createCIMPartitionPass()";
  // This pass creates cim and memref ops in modules that may contain
  // neither, so both dialects must be loaded before it runs.
  let dependentDialects = [
    "mlir::cim::CIMDialect",
    "mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"targetYAML", "target-yaml", "std::string", "\"\"",
           "Path to a target description YAML file (spec Sec. 7); supplies "
           "the tile geometry and per-op costs to partition against">
  ];
}

def CIMPlacement : Pass<"cim-placement", "ModuleOp"> {
  let summary = "Assign weight sub-matrices to physical tiles over time to "
                "minimize total cim.program cost (Belady / furthest-in-future eviction)";
  let description = [{
    Spec Sec. 6, Pass 3 — THE FLAGSHIP PASS. Given T physical tiles and W
    weight sub-matrices where W > T, and a static/known execution order,
    this has perfect future knowledge, making the theoretically optimal
    cache-replacement policy actually implementable: reuse a resident
    weight with no cim.program; otherwise assign a free tile; otherwise
    evict the resident weight whose NEXT USE is furthest in the future.
    For a model that fits entirely in tiles, this eliminates ALL
    reprogramming after the first pass.
  }];
  let constructor = "mlir::cim::createCIMPlacementPass()";
  let options = [
    Option<"targetYAML", "target-yaml", "std::string", "\"\"",
           "Path to a target description YAML file (spec Sec. 7); supplies "
           "tiles.count, the number of physical tiles to place into">
  ];
}

def CIMSchedule : Pass<"cim-schedule", "ModuleOp"> {
  let summary = "Order emitted cim ops, inserting barriers conservatively "
                "(v0.1) or overlapping program/mvm via double-buffering (v0.2+)";
  let constructor = "mlir::cim::createCIMSchedulePass()";
}

def CIMInsertTransfers : Pass<"cim-insert-transfers", "ModuleOp"> {
  let summary = "Insert cim.copy wherever an op needs an operand in a "
                "different #cim.space than it currently lives in";
  let description = [{
    Spec Sec. 6, Pass 5. Followed by a small dataflow analysis to hoist
    redundant copies out of loops.
  }];
  let constructor = "mlir::cim::createCIMInsertTransfersPass()";
}

def CIMLegalizePrecision : Pass<"cim-legalize-precision", "ModuleOp"> {
  let summary = "Insert cim.requantize after every partial-sum reduction "
                "per the target's declared output_effective_bits";
  let description = [{
    Spec Sec. 6, Pass 6. If the target declares effective_bits < 8, emit a
    warning with the estimated accuracy impact rather than silently
    degrading.
  }];
  let constructor = "mlir::cim::createCIMLegalizePrecisionPass()";
}

def CIMLowerToTarget : Pass<"cim-lower-to-target", "ModuleOp"> {
  let summary = "Lower cim ops to cimrt runtime calls for a concrete target";
  let description = [{
    Spec Sec. 6, Pass 7 (`--cim-lower-to-<target>` in the spec's naming; see
    file header note on why this is one parameterized pass here). For v0.1,
    lowers straight to cimrt runtime calls — a separate target dialect is
    premature until there are two real targets (spec M4).
  }];
  let constructor = "mlir::cim::createCIMLowerToTargetPass()";
  let options = [
    Option<"targetYAML", "target-yaml", "std::string", "\"\"",
           "Path to a target description YAML file (spec Sec. 7)">
  ];
}

def CIMCostReport : Pass<"cim-cost-report", "ModuleOp"> {
  let summary = "Walk the final IR, sum cost attributes, emit a JSON cost report";
  let description = [{
    Spec Sec. 6, Pass 8. Not an optimization pass — this is the pass that
    produces the project's publishable numbers (spec Sec. 17 worked example).

    Every cim.program and cim.mvm op is weighted by the product of its
    enclosing scf.for loops' compile-time-constant trip counts before being
    counted: a cim.program hoisted above a loop by cim-placement executes
    once regardless of position, while one left inside executes trip-count
    times, so a plain walk-and-count would misreport exactly the IR
    cim-placement now produces. An op under a loop whose trip count is not
    a compile-time constant is reported as unknown rather than assumed to
    be 1.
  }];
  let constructor = "mlir::cim::createCIMCostReportPass()";
  let options = [
    Option<"targetYAML", "target-yaml", "std::string", "\"\"",
           "Path to a target description YAML file (spec Sec. 7); supplies "
           "the cost table (energy_pj/latency_ns per program and mvm)">,
    Option<"outputPath", "output", "std::string", "\"\"",
           "Where to write the JSON report; empty (the default) means "
           "stdout">
  ];
}

#endif // CIM_TRANSFORMS_PASSES
