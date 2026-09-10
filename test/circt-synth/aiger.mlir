// RUN: circt-translate --export-aiger %s | circt-synth --format=aiger | FileCheck %s
// RUN: circt-translate --export-aiger %s -o %t.aig && circt-synth %t.aig | FileCheck %s
// CHECK-LABEL: @aiger_top(
// CHECK: synth.aig.and_inv %a, not %b : i1
hw.module @and(in %a: i1, in %b: i1, out and: i1) {
  %0 = synth.aig.and_inv %a, not %b : i1
  hw.output %0 : i1
}
