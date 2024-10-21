// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv

// TODO: Improve the error message
// expected-error @+1 {{initial ops cannot be topologically sorted}}
hw.module @toposort_failure(in %clk: !seq.clock, in %rst: i1, in %i: i32) {
  %init = seq.initial (%add) {
    ^bb0(%arg0: i32):
    seq.yield %arg0 : i32
  } : (!seq.immutable<i32>) -> !seq.immutable<i32>

  %add = seq.initial (%init) {
    ^bb0(%arg0 : i32):
    seq.yield %arg0 : i32
  } : (!seq.immutable<i32>) -> !seq.immutable<i32>

  %reg = seq.compreg %i, %clk initial %init : i32
  %reg2 = seq.compreg %i, %clk initial %add : i32
}

