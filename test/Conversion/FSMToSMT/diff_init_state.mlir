// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     [[F_A:%.+]] = smt.declare_fun "F_A" : !smt.func<(!smt.bv<2>) !smt.bool>
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
// CHECK-NEXT:     smt.assert [[FORALL0]]
// CHECK-NEXT:     [[FORALL1:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG2:%.+]]: !smt.bv<1>, [[ARG3:%.+]]: !smt.bv<1>, [[ARG4:%.+]]: !smt.bv<2>):
// CHECK-NEXT:       [[APP1:%.+]] = smt.apply_func [[F_A]]([[ARG4]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[ARG2]] : !smt.bv<1> to i1
// CHECK-NEXT:       [[CAST3:%.+]] = builtin.unrealized_conversion_cast [[ARG4]] : !smt.bv<2> to i2
// CHECK-NEXT:       [[APP2:%.+]] = smt.apply_func [[F_B]]([[ARG4]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       [[IMP1:%.+]] = smt.implies [[APP1]], [[APP2]]
// CHECK-NEXT:       smt.yield [[IMP1]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL1]]
// CHECK-NEXT:     [[FORALL2:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG5:%.+]]: !smt.bv<1>, [[ARG6:%.+]]: !smt.bv<1>, [[ARG7:%.+]]: !smt.bv<2>):
// CHECK-NEXT:       [[APP3:%.+]] = smt.apply_func [[F_B]]([[ARG7]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       [[CAST4:%.+]] = builtin.unrealized_conversion_cast [[ARG5]] : !smt.bv<1> to i1
// CHECK-NEXT:       [[CAST5:%.+]] = builtin.unrealized_conversion_cast [[ARG7]] : !smt.bv<2> to i2
// CHECK-NEXT:       [[APP4:%.+]] = smt.apply_func [[F_A]]([[ARG7]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       [[IMP2:%.+]] = smt.implies [[APP3]], [[APP4]]
// CHECK-NEXT:       smt.yield [[IMP2]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL2]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

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
