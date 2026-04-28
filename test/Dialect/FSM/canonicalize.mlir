// RUN: circt-opt --canonicalize %s | FileCheck %s

// CHECK-NOT: fsm.update

fsm.machine @foo(%arg0: i1) attributes {initialState = "A"} {
  %var = fsm.variable "var" {initValue = 0 : i16} : i16
  fsm.state @A transitions {
    fsm.transition @A action {
        fsm.update %var, %var : i16
    }
  }
}
