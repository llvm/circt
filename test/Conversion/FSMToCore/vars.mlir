// RUN: circt-opt -convert-fsm-to-core %s | FileCheck %s

// CHECK-LABEL: hw.module @has_var(
// CHECK-SAME: in [[CLK:%.+]] : !seq.clock, in [[RST:%.+]] : i1
// CHECK: [[FALSE:%.+]] = hw.constant false
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[STATE_REG:%.+]] = seq.compreg sym @state_reg
// CHECK: [[C42_I8:%.+]] = hw.constant 42 : i8
// CHECK: [[INIT:%.+]] = seq.initial() {
// CHECK-NEXT: [[C42_I8_0:%.+]] = hw.constant 42 : i8
// CHECK-NEXT: seq.yield [[C42_I8_0]] : i8
// CHECK: [[VAR1:%.+]] = seq.compreg sym @var1 [[NEXT_VAR1:%.+]], [[CLK]] reset [[RST]], [[C42_I8]] initial [[INIT]]
// CHECK: [[C1_I8:%.+]] = hw.constant 1 : i8
// CHECK: [[IN_A:%.+]] = comb.icmp eq %state_reg, %false : i1
// CHECK: [[VAR1PLUS1:%.+]] = comb.add [[VAR1]], [[C1_I8]] : i8
// CHECK: [[NEXT_VAR1]] = comb.mux [[IN_A]], [[VAR1PLUS1]], [[VAR1]] : i8

fsm.machine @has_var() -> () attributes {initialState = "A"} {
  %var1 = fsm.variable "var1" {initValue = 42 : i8} : i8
  %c1_i8 = hw.constant 1 : i8
  fsm.state @A output  {
  } transitions {
    fsm.transition @A action {
        %var1_next = comb.add %var1, %c1_i8 : i8
        fsm.update %var1, %var1_next : i8
    }
  }

  fsm.state @B output  {
  } transitions {
    fsm.transition @A action {
    }
  }
}
