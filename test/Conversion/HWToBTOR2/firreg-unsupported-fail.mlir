// An active-low reset or non-posedge clock on a seq.firreg is unsupported by the
// BTOR2 conversion. The rejection must FAIL the pass (non-zero exit), not merely
// emit a diagnostic and exit successfully with a partial model (state emitted but
// no `next` transition). `errors.mlir` checks these diagnostics with
// -verify-diagnostics, which masks the pass result; this test checks the failure.
// RUN: not circt-opt %s --convert-hw-to-btor2 --split-input-file -o /dev/null 2>&1 | FileCheck %s

// CHECK: active-low (NegReset) resets are not supported by the BTOR2 conversion
hw.module @ActiveLowResetFails(in %clk : !seq.clock, in %rst : i1, in %d : i32) {
  %c0 = hw.constant 0 : i32
  %r = seq.firreg %d clock %clk reset sync %rst, %c0 {clockEdge = 0 : i32, resetPolarity = 1 : i32} : i32
}

// -----

// CHECK: non-posedge clock edges are not supported by the BTOR2 conversion
hw.module @NegedgeClockFails(in %clk : !seq.clock, in %d : i32) {
  %r = seq.firreg %d clock %clk {clockEdge = 1 : i32} : i32
}
