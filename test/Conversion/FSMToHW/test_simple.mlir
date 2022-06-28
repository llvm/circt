
// RUN: circt-opt -convert-fsm-to-hw %s | FileCheck %s

// CHECK:     hw.module @top(%a0: i1, %a1: i1, %clk: i1, %rst: i1) -> (r0: i8) {
// CHECK-DAG:  %state_reg = seq.compreg %[[NEXT_STATE:.*]], %clk, %rst, %[[A_ENCODED:.*]]  : i2
// CHECK-DAG:  %A = sv.wire sym @A  : !hw.inout<i2>
// CHECK-DAG:  %[[A_ENCODING:.*]] = hw.constant 0 : i2
// CHECK-DAG:  sv.assign %A, %[[A_ENCODING]] : i2
// CHECK-DAG:  %[[A_ENCODED]] = sv.read_inout %A : !hw.inout<i2>
// CHECK-DAG:  %B = sv.wire sym @B  : !hw.inout<i2>
// CHECK-DAG:  %[[B_ENCODING:.*]] = hw.constant 1 : i2
// CHECK-DAG:  sv.assign %B, %[[B_ENCODING]] : i2
// CHECK-DAG:  %[[B_ENCODED:.*]]= sv.read_inout %B : !hw.inout<i2>
// CHECK-DAG:  %C = sv.wire sym @C  : !hw.inout<i2>
// CHECK-DAG:  %[[C_ENCODING:.*]] = hw.constant -2 : i2
// CHECK-DAG:  sv.assign %C, %[[C_ENCODING]] : i2
// CHECK-DAG:  %[[C_ENCODED:.*]] = sv.read_inout %C : !hw.inout<i2>
// CHECK-DAG:  %[[C0:.*]] = hw.constant 0 : i8
// CHECK-DAG:  %[[C1:.*]] = hw.constant 1 : i8
// CHECK-DAG:  %[[B_GUARD:.*]] = comb.and %a0, %a1 : i1
// CHECK-DAG:  %[[B_NEXT_STATE:.*]] = comb.mux %[[B_GUARD]], %[[C_ENCODED]], %[[B_ENCODED]]: i2
// CHECK-DAG:  %[[C2:.*]] = hw.constant 2 : i8
// CHECK-DAG:  %[[C_GUARD:.*]] = comb.and %a0, %a1 : i1
// CHECK-DAG:  %[[C_NEXT_STATE:.*]] = comb.mux %[[C_GUARD]], %[[A_ENCODED:.*]], %[[B_ENCODED]]: i2
// CHECK-DAG:  %[[NEXT_STATE_MUX:.*]] = hw.array_create %[[C_NEXT_STATE]], %[[B_NEXT_STATE]], %[[B_ENCODED]]{sv.namehint = "next_state_mux"} : i2
// CHECK-DAG:  %[[NEXT_STATE]] = hw.array_get %[[NEXT_STATE_MUX]][%state_reg] {sv.namehint = "state_next"} : !hw.array<3xi2>
// CHECK-DAG:  %[[OUT0_MUX:.*]] = hw.array_create %[[C2]], %[[C1]], %[[C0]] {sv.namehint = "output_0_mux"} : i8
// CHECK-DAG:  %[[OUT0:.*]] = hw.array_get %[[OUT0_MUX]][%state_reg] : !hw.array<3xi8>
// CHECK-DAG:  hw.output %[[OUT0]] : i8
// CHECK-DAG:}

fsm.machine @top(%a0: i1, %arg1: i1) -> (i8) attributes {initialState = "A", argNames = ["a0", "a1"], resNames = ["r0"]} {

  fsm.state "A" output  {
    %c_0 = hw.constant 0 : i8
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @B
  }

  fsm.state "B" output  {
    %c_1 = hw.constant 1 : i8
    fsm.output %c_1 : i8
  } transitions {
    fsm.transition @C guard {
      %g = comb.and %a0, %arg1 : i1
      fsm.return %g
    }
  }

  fsm.state "C" output  {
    %c_2 = hw.constant 2 : i8
    fsm.output %c_2 : i8
  } transitions {
    fsm.transition @A guard {
      %g = comb.and %a0, %arg1 : i1
      fsm.return %g
    }
    fsm.transition @B
  }
}
