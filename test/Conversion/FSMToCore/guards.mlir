// RUN: circt-opt -convert-fsm-to-core %s | FileCheck %s

// CHECK-LABEL: hw.module @guards(
// CHECK-SAME: in [[IN0:%.+]] : i1,
// CHECK: [[FALSE:%.+]] = hw.constant false
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[STATE_REG:%.+]] = seq.compreg sym @state_reg
// note: The code generates some redundant comparisons that it leaves for canonicalization to clean up.
// CHECK: [[IN_A1:%.+]] = comb.icmp eq [[STATE_REG]], [[FALSE]] : i1
// CHECK: [[IN_A2:%.+]] = comb.icmp eq [[STATE_REG]], [[FALSE]] : i1
// CHECK: [[MUX1:%.+]] = comb.mux [[IN_A2]], [[FALSE]], [[STATE_REG]] : i1
// CHECK: [[MUX2:%.+]] = comb.mux [[IN0]], [[TRUE]], [[FALSE]] : i1
// CHECK: [[AND1:%.+]] = comb.and [[IN0]], [[IN_A1]] : i1
// CHECK: [[NEXT1:%.+]] = comb.mux [[IN_A1]], [[MUX2]], [[MUX1]] : i1
// CHECK: [[IN_B1:%.+]] = comb.icmp eq [[STATE_REG]], [[TRUE]] : i1
// CHECK: [[IN_B2:%.+]] = comb.icmp eq [[STATE_REG]], [[TRUE]] : i1
// CHECK: [[MUX3:%.+]] = comb.mux [[IN_B2]], [[TRUE]], [[NEXT1]] : i1
// CHECK: [[MUX4:%.+]] = comb.mux [[IN0]], [[FALSE]], [[TRUE]] : i1
// CHECK: [[AND2:%.+]] = comb.and [[IN0]], [[IN_B1]] : i1
// CHECK: [[NEXT2:%.+]] = comb.mux [[IN_B1]], [[MUX4]], [[MUX3]] : i1
// CHECK-NEXT: hw.output

fsm.machine @guards(%arg0: i1) -> () attributes {initialState = "A"} {
  fsm.state @A output  {
  } transitions {
    fsm.transition @B guard {
        fsm.return %arg0
    }
  }

  fsm.state @B output  {
  } transitions {
    fsm.transition @A guard {
        fsm.return %arg0
    }
  }
}
