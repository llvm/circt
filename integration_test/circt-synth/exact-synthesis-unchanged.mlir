// REQUIRES: z3-integration
// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-exact-synthesis{allowed-ops=synth.aig.and_inv:2 max-area=1 sat-solver=z3})' | FileCheck %s

// A three-input parity function cannot be implemented by one binary AIG node.
// The pass must leave the truth table in place when the area limit is reached.
// CHECK-LABEL: hw.module @unsolved
// CHECK: %[[TABLE:.+]] = comb.truth_table %a, %b, %c ->
// CHECK: hw.output %[[TABLE]] : i1

hw.module @unsolved(in %a : i1, in %b : i1, in %c : i1, out y : i1) {
  %0 = comb.truth_table %a, %b, %c -> [false, true, true, false,
                                        true, false, false, true]
  hw.output %0 : i1
}
