// RUN: circt-opt --canonicalize %s | FileCheck %s

// Canonicalize unreachable states.

fsm.machine @foo(%arg0: i1) -> i1 attributes {initialState = "IDLE", stateType = i1} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state "IDLE" output  {
    fsm.output %arg0 : i1
  } transitions  {
    fsm.transition @B guard  {
      %c0_i16 = arith.constant 0 : i16
      %0 = arith.cmpi eq, %cnt, %c0_i16 : i16
      fsm.return %0
    } action  {
    }
  }

  // CHECK-NOT: fsm.state "A"
  fsm.state "A" output  {
    fsm.output %arg0 : i1
  } transitions  {}

  // CHECK: fsm.state "B"
  fsm.state "B" output  {
    fsm.output %arg0 : i1
  } transitions  {}

  // CHECK-NOT: fsm.state "C"
  fsm.state "C" output  {
    fsm.output %arg0 : i1
  } transitions  {}
}
