// RUN: circt-opt --split-input-file --canonicalize %s | FileCheck %s

// Check trivial transitions.

// CHECK-LABEL:   fsm.machine @foo(
// CHECK-SAME:                     %[[VAL_0:.*]]: i1) attributes {initialState = "A", stateType = i1} {
// CHECK:           %[[VAL_1:.*]] = fsm.variable "cnt" {initValue = 0 : i16} : i16
// CHECK:           fsm.state "A" output {
// CHECK:           } transitions {
// CHECK:             fsm.transition @C action {
// CHECK:               fsm.update %[[VAL_1]], %[[VAL_1]] : i16
// CHECK:             }
// CHECK:           }
// CHECK:           fsm.state "C" output {
// CHECK:           } transitions {
// CHECK:             fsm.transition @A action {
// CHECK:               fsm.update %[[VAL_1]], %[[VAL_1]] : i16
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK-NOT: fsm.state "B"
fsm.machine @foo(%arg0: i1) -> () attributes {initialState = "A", stateType = i1} {
  %cnt = fsm.variable "cnt" {initValue = 0 : i16} : i16

  fsm.state "A" output  {
  } transitions {
    fsm.transition @B action  {
      fsm.update %cnt, %cnt : i16
    }
  }

  fsm.state "B" output  {
  } transitions {
    fsm.transition @C
  }

  fsm.state "C" output  {
  } transitions {
    fsm.transition @A action {
      fsm.update %cnt, %cnt : i16
    }
  }
}
