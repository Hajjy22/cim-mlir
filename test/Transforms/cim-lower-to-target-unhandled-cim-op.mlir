// RUN: cim-opt %s --cim-lower-to-target=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file --verify-diagnostics

// A cim.* op this pass has no lowering for must be a LOUD, LOCATED error,
// not a silent pass-through.
//
// The pass rewrites everything around such an op into cimrt calls, so a
// survivor would be left dangling among lowered code and reach
// mlir-translate as an unknown dialect op -- failing far from its cause, or
// being silently dropped. The check is written against the dialect rather
// than a list of known-unhandled ops, so any op added to the cim dialect
// later inherits the same protection without anyone remembering this file.
//
// cim.reduce_max is the op that exposed the gap: the cim-run interpreter
// executes it (test/Run/reduce-max.mlir), but this compiled real-target
// path deliberately has no lowering for it yet, and "not supported here
// yet" has to be something a user can read.

func.func @unhandled_cim_op_is_refused_not_passed_through(
    %a: memref<1x2xi8>, %b: memref<1x2xi8>) -> memref<1x2xi8> {
  // expected-error @+1 {{cim-lower-to-target has no lowering for this op}}
  %m = cim.reduce_max %a, %b
     : (memref<1x2xi8>, memref<1x2xi8>) -> memref<1x2xi8>
  return %m : memref<1x2xi8>
}
