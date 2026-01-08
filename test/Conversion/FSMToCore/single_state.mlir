// RUN: circt-opt -convert-fsm-to-core %s | FileCheck %s

// CHECK-LABEL: hw.module @FSM(
// CHECK: [[C0_I8:%.+]] = hw.constant 0 : i8
// CHECK: hw.output [[C0_I8]] : i8

fsm.machine @FSM(%arg0: i1, %arg1: i1) -> (i8) attributes {initialState = "A"} {
  %c_0 = hw.constant 0 : i8
  fsm.state @A output  {
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @A
  }
}
