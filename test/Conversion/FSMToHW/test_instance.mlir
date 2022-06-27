// RUN: circt-opt -convert-fsm-to-hw %s | FileCheck %s

fsm.machine @FSM(%arg0: i1, %arg1: i1) -> (i8) attributes {initialState = "A"} {
  %c_0 = hw.constant 0 : i8
  fsm.state "A" output  {
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @B
  }

  fsm.state "B" output  {
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @A
  }
}

// CHECK: hw.module @top(%arg0: i1, %arg1: i1, %clk: i1, %rst: i1) -> (out: i8) {
// CHECK:   %fsm_inst.out0 = hw.instance "fsm_inst" @FSM(in0: %arg0: i1, in1: %arg1: i1, clk: %clk: i1, rst: %rst: i1) -> (out0: i8)
// CHECK:   hw.output %fsm_inst.out0 : i8
// CHECK: }
hw.module @top(%arg0: i1, %arg1: i1, %clk : i1, %rst : i1) -> (out: i8) {
    %out = fsm.hw_instance "fsm_inst" @FSM(%arg0, %arg1), clock %clk, reset %rst : (i1, i1) -> (i8)
    hw.output %out : i8
}
