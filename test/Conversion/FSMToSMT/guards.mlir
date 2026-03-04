// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT:  smt.solver() : () -> () {
// CHECK-NEXT:    [[C1_I2:%.+]] = hw.constant 1 : i2
// CHECK-NEXT:    [[C_1_I2:%.+]] = hw.constant -1 : i2
// CHECK-NEXT:    [[F_A:%.+]] = smt.declare_fun "F_A" : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:    [[F_B:%.+]] = smt.declare_fun "F_B" : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:    [[FORALL0:%.+]] = smt.forall {
// CHECK-NEXT:    ^bb0([[ARG0:%.+]]: !smt.bv<1>, [[ARG1:%.+]]: !smt.bv<2>):
// CHECK-NEXT:      [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : !smt.bv<1> to i1
// CHECK-NEXT:      [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[ARG1]] : !smt.bv<2> to i2
// CHECK-NEXT:      [[C0_I2:%.+]] = hw.constant 0 : i2
// CHECK-NEXT:      [[C0_BV2:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<2>
// CHECK-NEXT:      [[FUN0:%.+]] = smt.apply_func [[F_A]]([[C0_BV2]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      smt.yield [[FUN0]] : !smt.bool
// CHECK-NEXT:    } 
// CHECK-NEXT:    smt.assert [[FORALL0]]
// CHECK-NEXT:    [[FORALL1:%.+]] = smt.forall {
// CHECK-NEXT:    ^bb0([[ARG0_1:%.+]]: !smt.bv<1>, [[ARG1_1:%.+]]: !smt.bv<1>, [[ARG2_1:%.+]]: !smt.bv<2>):
// CHECK-NEXT:      [[FUN1:%.+]] = smt.apply_func [[F_A]]([[ARG2_1]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[ARG0_1]] : !smt.bv<1> to i1
// CHECK-NEXT:      [[CAST3:%.+]] = builtin.unrealized_conversion_cast [[ARG2_1]] : !smt.bv<2> to i2
// CHECK-NEXT:      [[CAST4:%.+]] = builtin.unrealized_conversion_cast [[C1_I2]] : i2 to !smt.bv<2>
// CHECK-NEXT:      [[FUN2:%.+]] = smt.apply_func [[F_B]]([[CAST4]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      [[CAST5:%.+]] = builtin.unrealized_conversion_cast [[ARG0_1]] : !smt.bv<1> to i1
// CHECK-NEXT:      [[CAST6:%.+]] = builtin.unrealized_conversion_cast [[ARG2_1]] : !smt.bv<2> to i2
// CHECK-NEXT:      [[CAST7:%.+]] = builtin.unrealized_conversion_cast [[CAST5]] : i1 to !smt.bv<1>
// CHECK-NEXT:      [[CAST8:%.+]] = builtin.unrealized_conversion_cast [[CAST7]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:      [[C_1_BV1:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:      [[EQ0:%.+]] = smt.eq [[CAST8]], [[C_1_BV1]] : !smt.bv<1>
// CHECK-NEXT:      [[AND0:%.+]] = smt.and [[FUN1]], [[EQ0]]
// CHECK-NEXT:      [[IMPLIES0:%.+]] = smt.implies [[AND0]], [[FUN2]]
// CHECK-NEXT:      smt.yield [[IMPLIES0]] : !smt.bool
// CHECK-NEXT:    }
// CHECK-NEXT:    smt.assert [[FORALL1]]
// CHECK-NEXT:    [[FORALL2:%.+]] = smt.forall {
// CHECK-NEXT:    ^bb0([[ARG0_2:%.+]]: !smt.bv<1>, [[ARG1_2:%.+]]: !smt.bv<1>, [[ARG2_2:%.+]]: !smt.bv<2>):
// CHECK-NEXT:      [[FUN3:%.+]] = smt.apply_func [[F_B]]([[ARG2_2]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      [[CAST9:%.+]] = builtin.unrealized_conversion_cast [[ARG0_2]] : !smt.bv<1> to i1
// CHECK-NEXT:      [[CAST10:%.+]] = builtin.unrealized_conversion_cast [[ARG2_2]] : !smt.bv<2> to i2
// CHECK-NEXT:      [[CAST11:%.+]] = builtin.unrealized_conversion_cast [[C_1_I2]] : i2 to !smt.bv<2>
// CHECK-NEXT:      [[FUN4:%.+]] = smt.apply_func [[F_A]]([[CAST11]]) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      [[CAST12:%.+]] = builtin.unrealized_conversion_cast [[ARG0_2]] : !smt.bv<1> to i1
// CHECK-NEXT:      [[CAST13:%.+]] = builtin.unrealized_conversion_cast [[ARG2_2]] : !smt.bv<2> to i2
// CHECK-NEXT:      [[CAST14:%.+]] = builtin.unrealized_conversion_cast [[CAST12]] : i1 to !smt.bv<1>
// CHECK-NEXT:      [[CAST15:%.+]] = builtin.unrealized_conversion_cast [[CAST14]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:      [[C_1_BV1_0:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:      [[EQ1:%.+]] = smt.eq [[CAST15]], [[C_1_BV1_0]] : !smt.bv<1>
// CHECK-NEXT:      [[AND1:%.+]] = smt.and [[FUN3]], [[EQ1]]
// CHECK-NEXT:      [[IMPLIES1:%.+]] = smt.implies [[AND1]], [[FUN4]]
// CHECK-NEXT:      smt.yield [[IMPLIES1]] : !smt.bool
// CHECK-NEXT:    }
// CHECK-NEXT:    smt.assert [[FORALL2]]
// CHECK-NEXT:  }
// CHECK-NEXT: }

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
