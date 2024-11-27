// RUN: circt-opt --lower-formal-to-hw %s | FileCheck %s

hw.module @Foo(in %bar : i32, in %baz : i16, in %clk : !seq.clock) {}

// CHECK-LABEL: hw.module @FormalTop(in %symbolic_value_0 : i32, in %symbolic_value_1 : i16)
verif.formal @FormalTop {} {
  %0 = verif.symbolic_value : i32
  %1 = verif.symbolic_value : i16
  // CHECK: [[CLK:%[0-9]+]] = seq.const_clock high
  %high = seq.const_clock high
  %2 = comb.extract %0 from 16 :  (i32) -> i16
  // CHECK: [[RES:%[0-9]+]] = comb.xor
  %3 = comb.xor %1, %2 : i16
  // CHECK: hw.instance "foo" @Foo(bar: %symbolic_value_0: i32, baz: [[RES]]: i16, clk: [[CLK]]: !seq.clock)
  hw.instance "foo" @Foo(bar: %0: i32, baz: %3: i16, clk: %high: !seq.clock) -> ()
}
