// RUN: circt-opt -convert-fsm-to-core %s | FileCheck %s

// CHECK-LABEL: hw.module @alternating(
// CHECK-SAME: out {{.+}} : i8
// CHECK-SAME: in [[CLK:%.+]] : !seq.clock
// CHECK-SAME: in [[RST:%.+]] : i1
// CHECK-NEXT: [[FALSE:%.+]] = hw.constant false
// CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
// CHECK-NEXT: [[INIT:%.+]] = seq.initial() {
// CHECK-NEXT:   [[FALSE_1:%.+]] = hw.constant false
// CHECK-NEXT:   seq.yield [[FALSE_1]] : i1
// CHECK: [[STATE_REG:%.+]] = seq.compreg sym @state_reg [[NEWSTATE:%.+]], [[CLK]] reset [[RST]], [[FALSE]] initial [[INIT]] : i1  
// CHECK-NEXT: [[C0_I8:%.+]] = hw.constant 0 : i8
// CHECK-NEXT: [[C1_I8:%.+]] = hw.constant 1 : i8
// CHECK-NEXT: [[CMP1:%.+]] = comb.icmp eq [[STATE_REG]], [[FALSE]] : i1
// CHECK-NEXT: [[CMP2:%.+]] = comb.icmp eq [[STATE_REG]], [[FALSE]] : i1
// CHECK-NEXT: [[MUX1:%.+]] = comb.mux [[CMP2]], [[TRUE]], [[STATE_REG]] : i1
// CHECK-NEXT: [[CMP3:%.+]] = comb.icmp eq [[STATE_REG]], [[TRUE]] : i1
// CHECK-NEXT: [[MUX2:%.+]] = comb.mux [[CMP3]], [[C1_I8]], [[C0_I8]] : i8
// CHECK-NEXT: [[CMP4:%.+]] = comb.icmp eq [[STATE_REG]], [[TRUE]] : i1
// CHECK-NEXT: [[NEWSTATE]] = comb.mux [[CMP4]], [[FALSE]], [[MUX1]] : i1
// CHECK-NEXT: hw.output [[MUX2]] : i8

// CHECK-LABEL: hw.module @top(
// CHECK-SAME: in [[TOPCLK:%.+]] : !seq.clock
// CHECK-SAME: in [[TOPRST:%.+]] : i1
// CHECK-SAME: out {{.+}} : i8
// CHECK-NEXT: [[FSM_INST:%.+]] = hw.instance "fsm_inst" @alternating(clk: [[TOPCLK]]: !seq.clock, rst: [[TOPRST]]: i1) -> (out0: i8)
// CHECK-NEXT: hw.output [[FSM_INST]] : i8

fsm.machine @alternating() -> (i8) attributes {initialState = "A"} {
  %c_0 = hw.constant 0 : i8
  %c_1 = hw.constant 1 : i8
  fsm.state @A output  {
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @B
  }

  fsm.state @B output  {
    fsm.output %c_1 : i8
  } transitions {
    fsm.transition @A
  }
}

hw.module @top(in %clk : !seq.clock, in %rst : i1, out out: i8) {
    %out = fsm.hw_instance "fsm_inst" @alternating(), clock %clk, reset %rst : () -> (i8)
    hw.output %out : i8
}
