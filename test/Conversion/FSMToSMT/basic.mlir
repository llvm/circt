// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT:  smt.solver() : () -> () {
// CHECK-NEXT:    %c0_i8 = hw.constant 0 : i8
// CHECK-NEXT:    %c1_i8 = hw.constant 1 : i8
// CHECK-NEXT:    %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:    %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:    %0 = smt.forall {
// CHECK-NEXT:    ^bb0(%arg0: !smt.bv<8>):
// CHECK-NEXT:      %3 = builtin.unrealized_conversion_cast %c0_i8 : i8 to !smt.bv<8>
// CHECK-NEXT:      %4 = smt.apply_func %F_A(%3) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:      smt.yield %4 : !smt.bool
// CHECK-NEXT:    }
// CHECK-NEXT:    smt.assert %0
// CHECK-NEXT:    %1 = smt.forall {
// CHECK-NEXT:    ^bb0(%arg0: !smt.bv<8>):
// CHECK-NEXT:      %3 = smt.apply_func %F_A(%arg0) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:      %4 = builtin.unrealized_conversion_cast %c1_i8 : i8 to !smt.bv<8>
// CHECK-NEXT:      %5 = smt.apply_func %F_B(%4) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:      %6 = smt.implies %3, %5
// CHECK-NEXT:      smt.yield %6 : !smt.bool
// CHECK-NEXT:    }
// CHECK-NEXT:    smt.assert %1
// CHECK-NEXT:    %2 = smt.forall {
// CHECK-NEXT:    ^bb0(%arg0: !smt.bv<8>):
// CHECK-NEXT:      %3 = smt.apply_func %F_B(%arg0) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:      %4 = builtin.unrealized_conversion_cast %c0_i8 : i8 to !smt.bv<8>
// CHECK-NEXT:      %5 = smt.apply_func %F_A(%4) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:      %6 = smt.implies %3, %5
// CHECK-NEXT:      smt.yield %6 : !smt.bool
// CHECK-NEXT:    }
// CHECK-NEXT:    smt.assert %2
// CHECK-NEXT:  }
// CHECK-NEXT: }

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
