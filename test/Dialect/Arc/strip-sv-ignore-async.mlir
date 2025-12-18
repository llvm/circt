// RUN: circt-opt %s --arc-strip-sv='async-resets-as-sync' | FileCheck %s

// CHECK: hw.module @AsyncReg(in [[CLK:%.+]] : !seq.clock, in [[RST:%.+]] : i1, in [[ARG0:%.+]] : i8) {
// CHECK:  [[C0_I8:%.+]] = hw.constant 0 : i8
// CHECK:  {{%.+}} = seq.compreg [[ARG0]], [[CLK]] reset [[RST]], [[C0_I8]] : i8

hw.module @AsyncReg(in %clk : !seq.clock, in %rst : i1, in %arg0: i8) {
  %c0_i8 = hw.constant 0 : i8
  %int_rtc_tick_value = seq.firreg %arg0 clock %clk reset async %rst, %c0_i8 : i8
}
