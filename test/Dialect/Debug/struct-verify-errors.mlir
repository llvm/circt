// RUN: circt-opt %s --verify-diagnostics --split-input-file

// `flips` needs one entry per field.

func.func @StructFlipsSize(%a: i32, %b: i32) {
  // expected-error @+1 {{has 1 flips for 2 fields}}
  %0 = dbg.struct {"a": %a, "b": %b} {flips = array<i1: true>} : i32, i32
  return
}

// -----

// A `flips` with no flipped field is left out.

func.func @StructFlipsNone(%a: i32, %b: i32) {
  // expected-error @+1 {{has no flipped field; leave out 'flips'}}
  %0 = dbg.struct {"a": %a, "b": %b} {flips = array<i1: false, false>} : i32, i32
  return
}
