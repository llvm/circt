// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// CHECK:     [[F_A:%.+]] = smt.declare_fun "F_A" : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:     [[F_B:%.+]] = smt.declare_fun "F_B" : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:     [[FORALL0:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG0:%.+]]: !smt.bv<1>, [[ARG1:%.+]]: !smt.bv<2>):
// CHECK-NEXT:       [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : !smt.bv<1> to i1
// CHECK-NEXT:       [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[ARG1]] : !smt.bv<2> to i2
// CHECK-NEXT:       [[C0_I2:%.+]] = hw.constant 0 : i2
// CHECK-NEXT:       [[C0_BV2:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<2>
// CHECK-NEXT:       [[APP0:%.+]] = smt.apply_func [[F_B]]([[C0_BV2]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       smt.yield [[APP0]] : !smt.bool
// CHECK-NEXT:     }


fsm.machine @guards(%arg0: i1) -> () attributes {initialState = "B"} {
  %var1 = fsm.variable "var1" {initValue = 0 : i2} : i2

  fsm.state @A output  {
  } transitions {
    fsm.transition @B 
  }

  fsm.state @B output  {
  } transitions {
    fsm.transition @A 
  }
}
