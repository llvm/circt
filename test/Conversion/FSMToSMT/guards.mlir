// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s
// CHECK: module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:     %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:     %0 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<2>):
// CHECK-NEXT:       %c0_bv2 = smt.bv.constant #smt.bv<0> : !smt.bv<2>
// CHECK-NEXT:       %3 = smt.apply_func %F_A(%c0_bv2) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       smt.yield %3 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %0
// CHECK-NEXT:     %1 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<2>):
// CHECK-NEXT:       %3 = smt.apply_func %F_A(%arg2) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       %c1_bv2 = smt.bv.constant #smt.bv<1> : !smt.bv<2>
// CHECK-NEXT:       %4 = smt.apply_func %F_B(%c1_bv2) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %5 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %6 = smt.and %3, %5
// CHECK-NEXT:       %7 = smt.implies %6, %4
// CHECK-NEXT:       smt.yield %7 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %1
// CHECK-NEXT:     %2 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<2>):
// CHECK-NEXT:       %3 = smt.apply_func %F_B(%arg2) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       %c-1_bv2 = smt.bv.constant #smt.bv<-1> : !smt.bv<2>
// CHECK-NEXT:       %4 = smt.apply_func %F_A(%c-1_bv2) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %5 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %6 = smt.and %3, %5
// CHECK-NEXT:       %7 = smt.implies %6, %4
// CHECK-NEXT:       smt.yield %7 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %2
// CHECK-NEXT:   }
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
