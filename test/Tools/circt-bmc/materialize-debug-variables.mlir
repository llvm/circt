// RUN: circt-opt --externalize-registers --materialize-debug-variables %s | FileCheck %s

// CHECK-LABEL: hw.module @named_regs(
// CHECK:         dbg.variable "in0", [[IN0:%.+]] : i32
// CHECK:         dbg.variable "in1", [[IN1:%.+]] : i32
// CHECK:         dbg.variable "firstreg", %firstreg_state : i32
// CHECK:         dbg.variable "secondreg", %secondreg_state : i32
// CHECK-NOT:     dbg.variable "clk"
hw.module @named_regs(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  %firstreg = seq.compreg %0, %clk : i32
  %secondreg = seq.compreg %firstreg, %clk : i32
  hw.output %secondreg : i32
}

// -----

// CHECK-LABEL: hw.module @preserve_existing_and_non_normalized(
// CHECK:         dbg.variable "y", [[Y:%.+]] : i8
// CHECK:         dbg.variable "data_state", [[STATE:%.+]] : i8
// CHECK:         dbg.variable "x_alias", [[X:%.+]] : i8
// CHECK-NOT:     dbg.variable "x", [[X]]
hw.module @preserve_existing_and_non_normalized(in %clk: !seq.clock, in %x: i8, in %y: i8, in %data_state: i8, out out: i8) {
  dbg.variable "x_alias", %x : i8
  %0 = comb.add %x, %y : i8
  hw.output %0 : i8
}
