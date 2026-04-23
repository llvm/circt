// RUN: circt-opt --materialize-debug-variables %s | FileCheck %s

// CHECK-LABEL: hw.module @named_regs(
// CHECK-SAME:   in [[CLK:%[^:]+]] : !seq.clock
// CHECK-SAME:   in [[IN0:%[^:]+]] : i32
// CHECK-SAME:   in [[IN1:%[^:]+]] : i32
// CHECK:         dbg.variable "in0", [[IN0]] : i32
// CHECK:         dbg.variable "in1", [[IN1]] : i32
// CHECK:         [[FIRSTREG:%.+]] = seq.compreg
// CHECK-NEXT:    dbg.variable "firstreg", [[FIRSTREG]] : i32
// CHECK:         [[SECONDREG:%.+]] = seq.compreg
// CHECK-NEXT:    dbg.variable "secondreg", [[SECONDREG]] : i32
// CHECK-NOT:     dbg.variable "clk", [[CLK]]
hw.module @named_regs(in %clk: !seq.clock, in %in0: i32, in %in1: i32, out out: i32) {
  %0 = comb.add %in0, %in1 : i32
  %firstreg = seq.compreg %0, %clk : i32
  %secondreg = seq.compreg %firstreg, %clk : i32
  hw.output %secondreg : i32
}

// -----

// CHECK-LABEL: hw.module @preserve_existing_and_non_normalized(
// CHECK-SAME:   in [[X_IN:%[^:]+]] : i8
// CHECK-SAME:   in [[Y_IN:%[^:]+]] : i8
// CHECK-SAME:   in [[DATA_STATE_IN:%[^:]+]] : i8
// CHECK:         dbg.variable "y", [[Y_IN]] : i8
// CHECK:         dbg.variable "data_state", [[DATA_STATE_IN]] : i8
// CHECK:         dbg.variable "x_alias", [[X_IN]] : i8
// CHECK-NOT:     dbg.variable "x", [[X_IN]]
hw.module @preserve_existing_and_non_normalized(in %clk: !seq.clock, in %x: i8, in %y: i8, in %data_state: i8, out out: i8) {
  dbg.variable "x_alias", %x : i8
  %0 = comb.add %x, %y : i8
  hw.output %0 : i8
}

// -----

// CHECK-LABEL: hw.module @do_not_guess_from_suffix(
// CHECK-SAME:   in [[FOO_STATE_IN:%[^:]+]] : i1
// CHECK:         dbg.variable "foo_state", [[FOO_STATE_IN]] : i1
// CHECK-NOT:     dbg.variable "foo", [[FOO_STATE_IN]]
hw.module @do_not_guess_from_suffix(in %foo_state: i1, out foo_next: i1) {
  hw.output %foo_state : i1
}
