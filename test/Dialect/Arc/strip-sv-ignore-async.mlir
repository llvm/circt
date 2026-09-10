// RUN: circt-opt %s --arc-strip-sv='async-resets-as-sync' | FileCheck %s

// CHECK: hw.module @AsyncReg(in [[CLK:%.+]] : !seq.clock, in [[RST:%.+]] : i1, in [[ARG0:%.+]] : i8) {
// CHECK:  [[C0_I8:%.+]] = hw.constant 0 : i8
// CHECK:  {{%.+}} = seq.compreg [[ARG0]], [[CLK]] reset [[RST]], [[C0_I8]] : i8

hw.module @AsyncReg(in %clk : !seq.clock, in %rst : i1, in %arg0: i8) {
  %c0_i8 = hw.constant 0 : i8
  %int_rtc_tick_value = seq.firreg %arg0 clock %clk reset async %rst, %c0_i8 : i8
}

// CHECK-LABEL: hw.module @HasBeenResetAsync(
hw.module @HasBeenResetAsync(in %clock: i1, in %reset: i1, out z: i1) {
  // CHECK-DAG: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK-DAG: [[INIT:%.+]] = seq.initial() {
  // CHECK-DAG:   [[FALSE:%.+]] = hw.constant false
  // CHECK-DAG:   seq.yield [[FALSE]]
  // CHECK-DAG: }
  // CHECK-DAG: [[TRUE:%.+]] = hw.constant true
  // CHECK-DAG: [[REG:%.+]] = seq.compreg [[REG]], [[CLK]] reset %reset, [[TRUE]] initial [[INIT]] : i1
  // CHECK-DAG: [[NOT_RESET:%.+]] = comb.xor %reset, %{{.+}} : i1
  // CHECK-DAG: [[OUT:%.+]] = comb.and bin [[REG]], [[NOT_RESET]] : i1
  %0 = verif.has_been_reset %clock, async %reset
  // CHECK: hw.output [[OUT]]
  hw.output %0 : i1
}
