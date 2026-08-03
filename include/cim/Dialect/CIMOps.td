//===- CIMOps.td - CIM dialect operations ------------------*- tablegen -*-===//
//
// Full op set from spec Section 5.3. `cim.program` and `cim.mvm` are given
// complete ODS (they are the two ops the whole project exists to optimize
// around and lower correctly — spec Sec. 3.3). The remaining seven are
// declared with summary + operand/result shape only; full semantics are
// `// TODO(spec Sec.5)` pending a first working lowering pipeline (M2).
//
// Verifier rules (spec Sec. 5.4), implemented in lib/Dialect/CIM/IR/CIMOps.cpp:
//   1. cim.mvm operand shapes must match the !cim.resident geometry exactly.
//   2. Activations to cim.mvm must be in #cim.space<near>. No implicit host access.
//   3. A !cim.resident may not be used after its tile has been re-programmed.
//      Enforced via SSA: cim.program returns a NEW resident handle. This is
//      the rule that catches real bugs — see CIM_ProgramOp below.
//   4. cim.reduce_partial operands must all have identical shape and be i32.
//   5. Tile IDs must be unique and within the target's declared tile count.
//
//===----------------------------------------------------------------------===//

#ifndef CIM_OPS
#define CIM_OPS

include "cim/Dialect/CIMDialect.td"
include "cim/Dialect/CIMAttrs.td"
include "cim/Dialect/CIMTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// Device and tile management
//===----------------------------------------------------------------------===//

def CIM_DeviceOpenOp : CIM_Op<"device_open", []> {
  let summary = "Open a target device by name, returning a device handle";
  // TODO(spec Sec.5.3): e.g. cim.device_open {target = "erbium-8t"} : !cim.device<"erbium-8t">
  let arguments = (ins StrAttr:$target);
  let results = (outs CIM_DeviceType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def CIM_TileAllocOp : CIM_Op<"tile_alloc", []> {
  let summary = "Allocate a physical tile on an open device";
  // TODO(spec Sec.5.3): e.g. cim.tile_alloc %dev {id = 0} : !cim.tile<256x256xi8>
  let arguments = (ins CIM_DeviceType:$device, I64Attr:$id);
  let results = (outs CIM_TileType:$result);
  let assemblyFormat = "$device attr-dict `:` type($result)";
}

def CIM_TileFreeOp : CIM_Op<"tile_free", []> {
  let summary = "Free a previously allocated tile";
  let arguments = (ins CIM_TileType:$tile);
  let assemblyFormat = "$tile attr-dict `:` type($tile)";
}

//===----------------------------------------------------------------------===//
// Weight programming - the expensive op (spec Sec. 3.3, 5.3)
//===----------------------------------------------------------------------===//

def CIM_ProgramOp : CIM_Op<"program", [
    MemoryEffects<[MemWrite]>
  ]> {
  let summary = "Program a weight sub-matrix into a tile";
  let description = [{
    This is the op the whole optimizer exists to minimize (spec Sec. 3.3):
    on non-volatile CIM hardware, reprogramming a tile is orders of
    magnitude more expensive than computing against it. Deliberately
    NOT foldable, hoistable, or CSE-able by default — it carries
    MemoryEffects<[MemWrite]> on the tile resource, because a generic pass
    "helpfully" optimizing away a weight load is a class of catastrophic
    miscompile this dialect exists to prevent.

    Verifier rule 3 (Sec. 5.4): the returned !cim.resident is a fresh SSA
    value and invalidates any prior resident handle for this tile. Weight
    residency is tracked through SSA use-def chains, not as a side effect
    on a mutable tile object — this is what lets cim-placement (Sec. 6)
    do Belady/furthest-in-future eviction analysis directly on the IR.

    Example:
      %r = cim.program %t, %weights {
             cost_ns = 12000 : i64, cost_pj = 480000 : i64, persistent = true
           } : (!cim.tile<256x256xi8>, memref<256x256xi8, #cim.space<host>>)
             -> !cim.resident<256x256xi8>
  }];
  let arguments = (ins
    CIM_TileType:$tile,
    AnyMemRef:$weights,
    I64Attr:$cost_ns,
    I64Attr:$cost_pj,
    DefaultValuedAttr<BoolAttr, "true">:$persistent
  );
  let results = (outs CIM_ResidentType:$result);
  let assemblyFormat = "$tile `,` $weights attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1; // TODO(rule 3): enforce prior-resident invalidation
}

//===----------------------------------------------------------------------===//
// Compute
//===----------------------------------------------------------------------===//

def CIM_MvmOp : CIM_Op<"mvm", []> {
  let summary = "Matrix-vector (or matrix-matrix) multiply against resident weights";
  let description = [{
    The fundamental CIM primitive. Output type is a wider integer type than
    the input (e.g. i32 accumulator for i8 weights/activations) so that
    partial sums are never silently saturated (spec Sec. 5.3) — requantize
    is an explicit later op (cim.requantize).

    Example:
      %out = cim.mvm %r, %act {accumulate = false}
             : (!cim.resident<256x256xi8>, memref<256xi8, #cim.space<near>>)
             -> memref<256xi32, #cim.space<near>>
  }];
  let arguments = (ins
    CIM_ResidentType:$weights,
    AnyMemRef:$activations,
    DefaultValuedAttr<BoolAttr, "false">:$accumulate
  );
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "$weights `,` $activations attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1; // TODO(rules 1-2): resident-geometry match, near-space check
}

//===----------------------------------------------------------------------===//
// Partial-sum reduction
//===----------------------------------------------------------------------===//

def CIM_ReducePartialOp : CIM_Op<"reduce_partial", []> {
  let summary = "Sum partial MVM outputs when a logical weight matrix spans multiple tiles";
  // TODO(spec Sec.5.3, rule 4): all operands must be identical shape and i32.
  let arguments = (ins Variadic<AnyMemRef>:$partials);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "$partials attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1; // TODO(rule 4)
}

//===----------------------------------------------------------------------===//
// Precision legalization
//===----------------------------------------------------------------------===//

def CIM_RequantizeOp : CIM_Op<"requantize", []> {
  let summary = "Requantize a partial sum to a tile-native precision (models digital requant or analog ADC readout)";
  // TODO(spec Sec.5.3): effective_bits < 8 models analog ADC accuracy loss explicitly.
  let arguments = (ins
    AnyMemRef:$input,
    F32Attr:$scale,
    I32Attr:$zero_point,
    I32Attr:$effective_bits
  );
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Data movement - always explicit (spec Sec. 3.4)
//===----------------------------------------------------------------------===//

def CIM_CopyOp : CIM_Op<"copy", []> {
  let summary = "Move a buffer between address spaces (#cim.space)";
  let description = [{
    Transfers between #cim.space<host|near|insitu> are always explicit ops,
    never implicit — this is the design decision that makes the cost model
    honest (spec Sec. 3.4). cim-insert-transfers (Sec. 6) is the pass that
    inserts these automatically where required.
  }];
  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def CIM_BarrierOp : CIM_Op<"barrier", []> {
  let summary = "Synchronize all outstanding cim ops on a device";
  let arguments = (ins CIM_DeviceType:$device);
  let assemblyFormat = "$device attr-dict `:` type($device)";
}

#endif // CIM_OPS
