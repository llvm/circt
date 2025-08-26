// RUN: circt-opt -convert-fsm-to-sv -lower-seq-to-sv -export-verilog %s -o /dev/null | FileCheck %s

// Smoke test to verify single state lowering to verilog is supported.
// CHECK: module top

fsm.machine @FSM(%arg0: i1, %arg1: i1) -> (i8) attributes {initialState = "A"} {
  %c_0 = hw.constant 0 : i8
  fsm.state @A output  {
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @A
  }
}

hw.module @top(in %arg0: i1, in %arg1: i1, in %clk : !seq.clock, in %rst : i1, out out: i8) {
    %out = fsm.hw_instance "fsm_inst" @FSM(%arg0, %arg1), clock %clk, reset %rst : (i1, i1) -> (i8)
    hw.output %out : i8
}

