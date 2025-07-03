// Run cp as a round-trip test.
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(aig-runner{solver-path="cp" solver-args="<inputFile>" solver-args="<outputFile>"}))' --verify-diagnostics

// expected-error@below {{multiple clocks found in the module}}
// expected-note@below {{previous clock is here}}
// expected-error@below {{failed to export module to AIGER format on module "multipleClock"}}
hw.module @multipleClock(in %c1: !seq.clock, in %c2: !seq.clock, in %a: i1, out out1: i1, out out2: i1) {
  %0 = seq.compreg %a, %c1 : i1
  %1 = seq.compreg %a, %c2 : i1
  hw.output %0, %1 : i1, i1
}

