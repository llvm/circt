// RUN: not circt-bmc %s -b 2 --module MultipleClocks --emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: error: modules with multiple clocks not yet supported
hw.module @MultipleClocks(in %clock0 : !seq.clock, in %clock1 : !seq.clock, in %a : i1, in %b : i1) {
  %0 = seq.from_clock %clock0
  %1 = seq.from_clock %clock1
  %2 = ltl.clocked_atom %a, posedge %0 : i1
  %3 = ltl.clocked_atom %b, posedge %1 : i1
  %4 = ltl.clocked_delay %3, posedge %1, 1, 0 : !ltl.sequence
  %5 = ltl.implication %2, %4 : !ltl.sequence, !ltl.sequence
  verif.assert %5 : !ltl.property
  hw.output
}
