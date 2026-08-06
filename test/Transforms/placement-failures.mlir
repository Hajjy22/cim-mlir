// RUN: not cim-opt %s --cim-placement --split-input-file 2>&1 \
// RUN:   | FileCheck --check-prefix=NO-TARGET %s
// RUN: not cim-opt %s --cim-placement=target-yaml=/nonexistent/target.yaml \
// RUN:   --split-input-file 2>&1 | FileCheck --check-prefix=BAD-PATH %s
// RUN: not cim-opt %s --cim-placement=target-yaml=%S/malformed-target.yaml \
// RUN:   --split-input-file 2>&1 | FileCheck --check-prefix=MALFORMED %s
// RUN: not cim-opt %s \
// RUN:   --cim-placement=target-yaml=%S/../targets/tiny-4x4.yaml \
// RUN:   --split-input-file 2>&1 | FileCheck --check-prefix=REWRITE %s

// cim-placement rewrites IR on the strength of a schedule. Every input it
// cannot reason about soundly has to stop the pass rather than be guessed
// at -- a wrong tile assignment produces IR that runs, prints numbers, and
// is silently computing against the wrong weights.

// The tile count to place into comes from the target file. Assuming one
// would produce a schedule for hardware that does not exist.
// NO-TARGET: requires -target-yaml
// BAD-PATH: failed to open target file
// MALFORMED: missing required field

// A weight memref whose offset is only known at runtime cannot be compared
// against another for equality, so the pass cannot tell whether two tiles
// hold the same weights. Guessing either way is wrong: guess "same" and it
// deletes a needed cim.program, guess "different" and it silently gives up
// the optimization it exists to perform.
// REWRITE: cannot identify a weight sub-matrix with a dynamic offset
func.func @dynamic_weight_offset(
    %t: !cim.tile<4x4xi8>,
    %w: memref<4x4xi8, strided<[4, 1], offset: ?>>) {
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
    : (!cim.tile<4x4xi8>, memref<4x4xi8, strided<[4, 1], offset: ?>>)
    -> !cim.resident<4x4xi8>
  return
}

// -----

// Assigning a tile id means writing it onto the cim.tile_alloc that
// produced the tile. Hand-written IR can hand a cim.program a tile from
// anywhere; there is nothing to assign to, so the pass says so instead of
// silently leaving the placement it computed unapplied.
// REWRITE: did not come from a cim.tile_alloc
func.func @tile_is_not_from_tile_alloc(%t: !cim.tile<4x4xi8>,
                                        %w: memref<4x4xi8>) {
  %r = cim.program %t, %w {cost_ns = 1 : i64, cost_pj = 1 : i64}
    : (!cim.tile<4x4xi8>, memref<4x4xi8>) -> !cim.resident<4x4xi8>
  return
}
