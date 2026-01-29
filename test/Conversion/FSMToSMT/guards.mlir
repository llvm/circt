// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s
// CHECK: module {
// CHECK-NEXT:  smt.solver() : () -> () {
// CHECK-NEXT:    %c1_i2 = hw.constant 1 : i2
// CHECK-NEXT:    %c-1_i2 = hw.constant -1 : i2
// CHECK-NEXT:    %F_A = smt.declare_fun "F_A" : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:    %F_B = smt.declare_fun "F_B" : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:    %0 = smt.forall {
// CHECK-NEXT:    ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<2>):
// CHECK-NEXT:      %3 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to i1
// CHECK-NEXT:      %4 = builtin.unrealized_conversion_cast %arg1 : !smt.bv<2> to i2
// CHECK-NEXT:      %c0_bv2 = smt.bv.constant #smt.bv<0> : !smt.bv<2>
// CHECK-NEXT:      %5 = smt.apply_func %F_A(%c0_bv2) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      smt.yield %5 : !smt.bool
// CHECK-NEXT:    }
// CHECK-NEXT:    smt.assert %0
// CHECK-NEXT:    %1 = smt.forall {
// CHECK-NEXT:    ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<2>):
// CHECK-NEXT:      %3 = smt.apply_func %F_A(%arg2) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      %4 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to i1
// CHECK-NEXT:      %5 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<2> to i2
// CHECK-NEXT:      %6 = builtin.unrealized_conversion_cast %c1_i2 : i2 to !smt.bv<2>
// CHECK-NEXT:      %7 = smt.apply_func %F_B(%6) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      %8 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to i1
// CHECK-NEXT:      %9 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<2> to i2
// CHECK-NEXT:      %10 = builtin.unrealized_conversion_cast %8 : i1 to !smt.bv<1>
// CHECK-NEXT:      %11 = builtin.unrealized_conversion_cast %10 : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:      %12 = smt.eq %11, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:      %13 = smt.and %3, %12
// CHECK-NEXT:      %14 = smt.implies %13, %7
// CHECK-NEXT:      smt.yield %14 : !smt.bool
// CHECK-NEXT:    }
// CHECK-NEXT:    smt.assert %1
// CHECK-NEXT:    %2 = smt.forall {
// CHECK-NEXT:    ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<2>):
// CHECK-NEXT:      %3 = smt.apply_func %F_B(%arg2) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      %4 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to i1
// CHECK-NEXT:      %5 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<2> to i2
// CHECK-NEXT:      %6 = builtin.unrealized_conversion_cast %c-1_i2 : i2 to !smt.bv<2>
// CHECK-NEXT:      %7 = smt.apply_func %F_A(%6) : !smt.func<(!smt.bv<2>) !smt.bool>
// CHECK-NEXT:      %8 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to i1
// CHECK-NEXT:      %9 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<2> to i2
// CHECK-NEXT:      %10 = builtin.unrealized_conversion_cast %8 : i1 to !smt.bv<1>
// CHECK-NEXT:      %11 = builtin.unrealized_conversion_cast %10 : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:      %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:      %12 = smt.eq %11, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:      %13 = smt.and %3, %12
// CHECK-NEXT:      %14 = smt.implies %13, %7
// CHECK-NEXT:      smt.yield %14 : !smt.bool
// CHECK-NEXT:    }
// CHECK-NEXT:    smt.assert %2
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
