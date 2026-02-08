// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s --check-prefix=NO-TIME
// RUN: circt-opt --convert-fsm-to-smt="with-time=true" %s | FileCheck %s -check-prefix=TIME


fsm.machine @alternating() -> (i8) attributes {initialState = "A"} {
  %c_0 = hw.constant 0 : i8
  %c_1 = hw.constant 1 : i8
  fsm.state @A output  {
    fsm.output %c_0 : i8
  } transitions {
    fsm.transition @B
  }

  fsm.state @B output  {
    fsm.output %c_1 : i8
  } transitions {
    fsm.transition @A
  }
}

// NO-TIME: module {
// NO-TIME:  smt.solver() : () -> () {
// NO-TIME:    [[C0_I8:%.+]] = hw.constant 0 : i8
// NO-TIME:    [[C1_I8:%.+]] = hw.constant 1 : i8
// NO-TIME:    [[F_A:%.+]] = smt.declare_fun "F_A" : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:    [[F_B:%.+]] = smt.declare_fun "F_B" : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:    [[FORALL0:%.+]] = smt.forall {
// NO-TIME:    ^bb0([[ARG0:%.+]]: !smt.bv<8>):
// NO-TIME:      [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[C0_I8]] : i8 to !smt.bv<8>
// NO-TIME:      [[FUN0:%.+]] = smt.apply_func [[F_A]]([[CAST0]]) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      smt.yield [[FUN0]] : !smt.bool
// NO-TIME:    }
// NO-TIME:    smt.assert [[FORALL0]]
// NO-TIME:    [[FORALL1:%.+]] = smt.forall {
// NO-TIME:    ^bb0([[ARG0_1:%.+]]: !smt.bv<8>):
// NO-TIME:      [[FUN1:%.+]] = smt.apply_func [[F_A]]([[ARG0_1]]) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[C1_I8]] : i8 to !smt.bv<8>
// NO-TIME:      [[FUN2:%.+]] = smt.apply_func [[F_B]]([[CAST1]]) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      [[IMPLIES0:%.+]] = smt.implies [[FUN1]], [[FUN2]]
// NO-TIME:      smt.yield [[IMPLIES0]] : !smt.bool
// NO-TIME:    }
// NO-TIME:    smt.assert [[FORALL1]]
// NO-TIME:    [[FORALL2:%.+]] = smt.forall {
// NO-TIME:    ^bb0([[ARG0_2:%.+]]: !smt.bv<8>):
// NO-TIME:      [[FUN3:%.+]] = smt.apply_func [[F_B]]([[ARG0_2]]) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[C0_I8]] : i8 to !smt.bv<8>
// NO-TIME:      [[FUN4:%.+]] = smt.apply_func [[F_A]]([[CAST2]]) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      [[IMPLIES1:%.+]] = smt.implies [[FUN3]], [[FUN4]]
// NO-TIME:      smt.yield [[IMPLIES1]] : !smt.bool
// NO-TIME:    }
// NO-TIME:    smt.assert [[FORALL2]]
// NO-TIME:  }
// NO-TIME: }

// TIME: module {
// TIME:   smt.solver() : () -> () {
// TIME:     [[C0_I8:%.+]] = hw.constant 0 : i8
// TIME:     [[C1_I8:%.+]] = hw.constant 1 : i8
// TIME:     [[F_A:%.+]] = smt.declare_fun "F_A" : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.bool>
// TIME:     [[F_B:%.+]] = smt.declare_fun "F_B" : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.bool>
// TIME:     [[FORALL0:%.+]] = smt.forall {
// TIME:     ^bb0([[ARG0:%.+]]: !smt.bv<8>, [[ARG1:%.+]]: !smt.bv<5>):
// TIME:       [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[C0_I8]] : i8 to !smt.bv<8>
// TIME:       [[C0_BV5:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<5>
// TIME:       [[FUN0:%.+]] = smt.apply_func [[F_A]]([[CAST0]], [[C0_BV5]]) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.
// TIME: bool>
// TIME:       smt.yield [[FUN0]] : !smt.bool
// TIME:     }
// TIME:     smt.assert [[FORALL0]]
// TIME:     [[FORALL1:%.+]] = smt.forall {
// TIME:     ^bb0([[ARG0_1:%.+]]: !smt.bv<8>, [[ARG1_1:%.+]]: !smt.bv<5>):
// TIME:       [[FUN1:%.+]] = smt.apply_func [[F_A]]([[ARG0_1]], [[ARG1_1]]) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.
// TIME: bool>
// TIME:       [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[C1_I8]] : i8 to !smt.bv<8>
// TIME:       [[C1_BV5:%.+]] = smt.bv.constant #smt.bv<1> : !smt.bv<5>
// TIME:       [[ADD0:%.+]] = smt.bv.add [[ARG1_1]], [[C1_BV5]] : !smt.bv<5>
// TIME:       [[FUN2:%.+]] = smt.apply_func [[F_B]]([[CAST1]], [[ADD0]]) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.bool>
// TIME:       [[IMPLIES0:%.+]] = smt.implies [[FUN1]], [[FUN2]]
// TIME:       smt.yield [[IMPLIES0]] : !smt.bool
// TIME:     }
// TIME:     smt.assert [[FORALL1]]
// TIME:     [[FORALL2:%.+]] = smt.forall {
// TIME:     ^bb0([[ARG0_2:%.+]]: !smt.bv<8>, [[ARG1_2:%.+]]: !smt.bv<5>):
// TIME:       [[FUN3:%.+]] = smt.apply_func [[F_B]]([[ARG0_2]], [[ARG1_2]]) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.
// TIME: bool>
// TIME:       [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[C0_I8]] : i8 to !smt.bv<8>
// TIME:       [[C1_BV5_0:%.+]] = smt.bv.constant #smt.bv<1> : !smt.bv<5>
// TIME:       [[ADD1:%.+]] = smt.bv.add [[ARG1_2]], [[C1_BV5_0]] : !smt.bv<5>
// TIME:       [[FUN4:%.+]] = smt.apply_func [[F_A]]([[CAST2]], [[ADD1]]) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.bool>
// TIME:       [[IMPLIES1:%.+]] = smt.implies [[FUN3]], [[FUN4]]
// TIME:       smt.yield [[IMPLIES1]] : !smt.bool
// TIME:     }
// TIME:     smt.assert [[FORALL2]]
// TIME:   }
// TIME: }
