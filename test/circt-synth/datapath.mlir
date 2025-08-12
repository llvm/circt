// RUN: circt-synth %s | FileCheck %s
// RUN: circt-synth %s --disable-datapath | FileCheck %s

// CHECK-LABEL: @arith
hw.module @arith(in %a: i4, in %b: i4, in %c: i4, out add: i4, out mul: i4) {
  %0 = comb.add %a, %b, %c : i4
  %1 = comb.mul %a, %b : i4
  hw.output %0, %1 : i4, i4
}
