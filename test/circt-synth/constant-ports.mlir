// RUN: circt-synth %s --enable-parameterize-constant-ports -output-longest-path=- -top Top  | FileCheck %s
// RUN: circt-synth %s -output-longest-path=- -top Top  | FileCheck %s --check-prefix CHECK-NO-PARAM

// CHECK: Maximum path delay: 9
// CHECK-NO-PARAM: Maximum path delay: 10

// CHECK: hw.param.value
hw.module private @MultipleConstantPorts(in %a: i4, in %b: i4, in %c: i4, out out: i4) {
  // a and b are both compile-time constants
  %0 = comb.add %a, %b : i4
  %1 = comb.mul %0, %c : i4
  hw.output %1 : i4
}

// CHECK-LABEL: hw.module @Top
hw.module @Top(in %data: i4, in %mode: i4, out out0: i4, out out1: i4) {
  %c0 = hw.constant 0 : i4
  %inst0.out = hw.instance "inst0" @MultipleConstantPorts(a: %c0: i4, b: %c0: i4, c: %data: i4) -> (out: i4)
  %c1 = hw.constant 1 : i4
  %inst1.out = hw.instance "inst1" @MultipleConstantPorts(a: %c1: i4, b: %c1: i4, c: %data: i4) -> (out: i4)
  hw.output %inst0.out, %inst1.out : i4, i4
}
