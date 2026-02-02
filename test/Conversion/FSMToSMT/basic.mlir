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
// NO-TIME:    %c0_i8 = hw.constant 0 : i8
// NO-TIME:    %c1_i8 = hw.constant 1 : i8
// NO-TIME:    %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:    %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:    %0 = smt.forall {
// NO-TIME:    ^bb0(%arg0: !smt.bv<8>):
// NO-TIME:      %3 = builtin.unrealized_conversion_cast %c0_i8 : i8 to !smt.bv<8>
// NO-TIME:      %4 = smt.apply_func %F_A(%3) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      smt.yield %4 : !smt.bool
// NO-TIME:    }
// NO-TIME:    smt.assert %0
// NO-TIME:    %1 = smt.forall {
// NO-TIME:    ^bb0(%arg0: !smt.bv<8>):
// NO-TIME:      %3 = smt.apply_func %F_A(%arg0) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      %4 = builtin.unrealized_conversion_cast %c1_i8 : i8 to !smt.bv<8>
// NO-TIME:      %5 = smt.apply_func %F_B(%4) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      %6 = smt.implies %3, %5
// NO-TIME:      smt.yield %6 : !smt.bool
// NO-TIME:    }
// NO-TIME:    smt.assert %1
// NO-TIME:    %2 = smt.forall {
// NO-TIME:    ^bb0(%arg0: !smt.bv<8>):
// NO-TIME:      %3 = smt.apply_func %F_B(%arg0) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      %4 = builtin.unrealized_conversion_cast %c0_i8 : i8 to !smt.bv<8>
// NO-TIME:      %5 = smt.apply_func %F_A(%4) : !smt.func<(!smt.bv<8>) !smt.bool>
// NO-TIME:      %6 = smt.implies %3, %5
// NO-TIME:      smt.yield %6 : !smt.bool
// NO-TIME:    }
// NO-TIME:    smt.assert %2
// NO-TIME:  }
// NO-TIME: }

// TIME: module {
// TIME:   smt.solver() : () -> () {
// TIME:     %c0_i8 = hw.constant 0 : i8
// TIME:     %c1_i8 = hw.constant 1 : i8
// TIME:     %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.bool>
// TIME:     %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.bool>
// TIME:     %0 = smt.forall {
// TIME:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<5>):
// TIME:       %3 = builtin.unrealized_conversion_cast %c0_i8 : i8 to !smt.bv<8>
// TIME:       %c0_bv5 = smt.bv.constant #smt.bv<0> : !smt.bv<5>
// TIME:       %4 = smt.apply_func %F_A(%3, %c0_bv5) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.
// TIME: bool>
// TIME:       smt.yield %4 : !smt.bool
// TIME:     }
// TIME:     smt.assert %0
// TIME:     %1 = smt.forall {
// TIME:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<5>):
// TIME:       %3 = smt.apply_func %F_A(%arg0, %arg1) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.
// TIME: bool>
// TIME:       %4 = builtin.unrealized_conversion_cast %c1_i8 : i8 to !smt.bv<8>
// TIME:       %c1_bv5 = smt.bv.constant #smt.bv<1> : !smt.bv<5>
// TIME:       %5 = smt.bv.add %arg1, %c1_bv5 : !smt.bv<5>
// TIME:       %6 = smt.apply_func %F_B(%4, %5) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.bool>
// TIME:       %7 = smt.implies %3, %6
// TIME:       smt.yield %7 : !smt.bool
// TIME:     }
// TIME:     smt.assert %1
// TIME:     %2 = smt.forall {
// TIME:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<5>):
// TIME:       %3 = smt.apply_func %F_B(%arg0, %arg1) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.
// TIME: bool>
// TIME:       %4 = builtin.unrealized_conversion_cast %c0_i8 : i8 to !smt.bv<8>
// TIME:       %c1_bv5 = smt.bv.constant #smt.bv<1> : !smt.bv<5>
// TIME:       %5 = smt.bv.add %arg1, %c1_bv5 : !smt.bv<5>
// TIME:       %6 = smt.apply_func %F_A(%4, %5) : !smt.func<(!smt.bv<8>, !smt.bv<5>) !smt.bool>
// TIME:       %7 = smt.implies %3, %6
// TIME:       smt.yield %7 : !smt.bool
// TIME:     }
// TIME:     smt.assert %2
// TIME:   }
// TIME: }