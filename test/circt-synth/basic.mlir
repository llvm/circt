// RUN: circt-synth %s | FileCheck %s

// CHECK-LABEL: @and
hw.module @and(in %a: i1, in %b: i1, out and: i1) {
  %0 = comb.and %a, %b : i1
  hw.output %0 : i1
}
