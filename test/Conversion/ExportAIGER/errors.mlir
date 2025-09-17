// RUN: circt-translate --export-aiger %s --split-input-file --verify-diagnostics

// Test unsupported variadic AND gates (should be lowered first)
hw.module @variadic_and_error(in %a: i1, in %b: i1, in %c: i1, out result: i1) {
  // expected-error @below {{variadic AND gates not supported, run aig-lower-variadic pass first}}
  %0 = synth.aig.and_inv %a, %b, %c : i1
  hw.output %0 : i1
}

// -----

// Test multiple clock signals (not supported)
// expected-error @below {{multiple clocks found in the module}}
// expected-note @below {{previous clock is here}}
hw.module @multiple_clocks_error(in %clk1: !seq.clock, in %clk2: !seq.clock, in %input: i1, out output: i1) {
  %reg1 = seq.compreg %input, %clk1 : i1
  %reg2 = seq.compreg %reg1, %clk2 : i1
  hw.output %reg2 : i1
}

// -----

// Test unsupported operation (when handleUnknownOperation is false)
hw.module @unknown_operation_error(in %a: i1, in %b: i1, out result: i1) {
  // expected-error @below {{unhandled operation}}
  %0 = comb.add %a, %b : i1
  hw.output %0 : i1
}

// -----

// Test graph that is not possible to topo-sort
// expected-error @below {{failed to sort operations topologically}}
hw.module @unsorted_latches_error(in %input1: i1, in %input2: i1, out output: i1) {
  %0 = synth.aig.and_inv %input1, %1 : i1
  %1 = synth.aig.and_inv %input1, %2 : i1
  %2 = synth.aig.and_inv %input1, %0 : i1
  hw.output %2 : i1
}
