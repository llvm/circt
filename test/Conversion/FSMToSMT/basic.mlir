// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s
// CHECK: module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     %c0_bv8 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:     %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:     %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:     %0 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>):
// CHECK-NEXT:       %c0_bv8_0 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:       %3 = smt.apply_func %F_A(%c0_bv8_0) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:       smt.yield %3 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %0
// CHECK-NEXT:     %1 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>):
// CHECK-NEXT:       %3 = smt.apply_func %F_A(%arg0) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:   §    %c1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:       %4 = smt.apply_func %F_B(%c1_bv8) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:       %true = smt.constant true
// CHECK-NEXT:       %5 = smt.and %3, %true
// CHECK-NEXT:       %6 = smt.implies %5, %4
// CHECK-NEXT:       smt.yield %6 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %1
// CHECK-NEXT:     %2 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>):
// CHECK-NEXT:       %3 = smt.apply_func %F_B(%arg0) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:       %c0_bv8_0 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:       %4 = smt.apply_func %F_A(%c0_bv8_0) : !smt.func<(!smt.bv<8>) !smt.bool>
// CHECK-NEXT:       %true = smt.constant true
// CHECK-NEXT:       %5 = smt.and %3, %true
// CHECK-NEXT:       %6 = smt.implies %5, %4
// CHECK-NEXT:       smt.yield %6 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %2
// CHECK-NEXT:   }
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
