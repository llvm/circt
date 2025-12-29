// RUN: circt-opt -convert-fsm-to-core %s | FileCheck %s

// CHECK-LABEL: hw.module @guards(
// CHECK-SAME: in [[IN0:%.+]] : i1,
// CHECK: [[FALSE:%.+]] = hw.constant false
// CHECK: [[TRUE:%.+]] = hw.constant true
// Use reset keyword as delimiter to avoid parsing whole rest of line
// CHECK: [[STATE_REG:%.+]] = seq.compreg sym @state_reg [[NEXT_STATE:%.+]], {{.*}} reset
// CHECK: [[C0_I2:%.+]] = hw.constant 0 : i2
// CHECK: [[VAR1:%.+]] = seq.compreg sym @var1 [[NEXT_VAR1:%.+]], {{.*}} reset
// CHECK: [[C1_I2:%.+]] = hw.constant 1 : i2
// CHECK: [[C3_I2:%.+]] = hw.constant -1 : i2
// note: The code generates some redundant comparisons that it leaves for canonicalization to clean up.
// CHECK: [[IN_A1:%.+]] = comb.icmp eq [[STATE_REG]], [[FALSE]] : i1
// CHECK: [[IN_A2:%.+]] = comb.icmp eq [[STATE_REG]], [[FALSE]] : i1
// CHECK: [[MUX1:%.+]] = comb.mux [[IN_A2]], [[FALSE]], [[STATE_REG]] : i1
// CHECK: [[MUX2:%.+]] = comb.mux [[IN0]], [[TRUE]], [[FALSE]] : i1
// CHECK: [[AND1:%.+]] = comb.and [[IN0]], [[IN_A1]] : i1
// CHECK: [[MUX3:%.+]] = comb.mux [[AND1]], [[C1_I2]], [[VAR1]] : i2
// CHECK: [[MUX4:%.+]] = comb.mux [[IN_A1]],
// CHECK: [[IN_B1:%.+]] = comb.icmp eq [[STATE_REG]], [[TRUE]] : i1
// CHECK: [[IN_B2:%.+]] = comb.icmp eq [[STATE_REG]], [[TRUE]] : i1
// CHECK: [[MUX5:%.+]] = comb.mux [[IN_B2]], [[TRUE]], [[MUX4]] : i1
// CHECK: [[MUX6:%.+]] = comb.mux [[IN0]], [[FALSE]], [[TRUE]] : i1
// CHECK: [[AND2:%.+]] = comb.and [[IN0]], [[IN_B1]] : i1
// CHECK: [[NEXT_VAR1]] = comb.mux [[AND2]], [[C3_I2]], [[MUX3]] : i2
// CHECK: [[NEXT_STATE]] = comb.mux [[IN_B1]], [[MUX6]], [[MUX5]] : i1
// CHECK-NEXT: hw.output

fsm.machine @guards(%arg0: i1) -> () attributes {initialState = "A"} {
  %var1 = fsm.variable "var1" {initValue = 0 : i2} : i2
  %c1_i2 = hw.constant 1 : i2
  %c3_i2 = hw.constant 3 : i2

  fsm.state @A output  {
  } transitions {
    fsm.transition @B guard {
        fsm.return %arg0
    } action {
        fsm.update %var1, %c1_i2 : i2
    }
  }

  fsm.state @B output  {
  } transitions {
    fsm.transition @A guard {
        fsm.return %arg0
    } action {
        fsm.update %var1, %c3_i2 : i2
    }
  }
}
