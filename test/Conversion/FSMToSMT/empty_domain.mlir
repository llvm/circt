// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// A machine with no outputs or variables should produce state functions with
// an empty domain

// CHECK: smt.declare_fun "F_A" : !smt.func<() !smt.bool>

fsm.machine @action() -> () attributes {initialState = "A"} {
  fsm.state @A output  {
  }
}
