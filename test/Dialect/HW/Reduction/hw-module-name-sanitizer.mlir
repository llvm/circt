// UNSUPPORTED: system-windows
// RUN: circt-reduce %s --include=hw-module-name-sanitizer --test /usr/bin/env --test-arg true --keep-best=0 | FileCheck %s

// CHECK-LABEL: hw.module @Foo(in %a : i8, in %b : i8, out c : i8)
// CHECK-LABEL: hw.module @Bar(in %a : i8, in %b : i8, out c : i8)
// CHECK: hw.instance "Foo" @Foo(a: %a: i8, b: %b: i8) -> (c: i8)

hw.module @Module1(in %input1: i8, in %input2: i8, out result: i8) {
  %0 = comb.xor %input1, %input2 : i8
  hw.output %0 : i8
}

hw.module @Module2(in %in1: i8, in %in2: i8, out out: i8) {
  %result = hw.instance "inst" @Module1(input1: %in1: i8, input2: %in2: i8) -> (result: i8)
  hw.output %result : i8
}
