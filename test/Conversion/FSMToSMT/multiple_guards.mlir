// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     %false = hw.constant false
// CHECK-NEXT:     %true = hw.constant true
// CHECK-NEXT:     %F_start = smt.declare_fun "F_start" : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:     %F_fail = smt.declare_fun "F_fail" : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:     %0 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>):
// CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %false : i1 to !smt.bv<1>
// CHECK-NEXT:       %5 = smt.apply_func %F_start(%4) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       smt.yield %5 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %0
// CHECK-NEXT:     %1 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>):
// CHECK-NEXT:       %4 = smt.apply_func %F_start(%arg0) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %false : i1 to !smt.bv<1>
// CHECK-NEXT:       %6 = smt.apply_func %F_start(%5) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %true : i1 to !smt.bv<1>
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %7 : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %9 = smt.eq %8, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %10 = smt.and %4, %9
// CHECK-NEXT:       %11 = smt.implies %10, %6
// CHECK-NEXT:       smt.yield %11 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %1
// CHECK-NEXT:     %2 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>):
// CHECK-NEXT:       %4 = smt.apply_func %F_start(%arg0) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %false : i1 to !smt.bv<1>
// CHECK-NEXT:       %6 = smt.apply_func %F_fail(%5) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %true : i1 to !smt.bv<1>
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %7 : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %9 = smt.eq %8, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %10 = builtin.unrealized_conversion_cast %true : i1 to !smt.bv<1>
// CHECK-NEXT:       %11 = builtin.unrealized_conversion_cast %10 : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %12 = smt.eq %11, %c-1_bv1_0 : !smt.bv<1>
// CHECK-NEXT:       %13 = smt.not %12
// CHECK-NEXT:       %14 = smt.and %9, %13
// CHECK-NEXT:       %15 = smt.and %4, %14
// CHECK-NEXT:       %16 = smt.implies %15, %6
// CHECK-NEXT:       smt.yield %16 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %2
// CHECK-NEXT:     %3 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>):
// CHECK-NEXT:       %4 = smt.apply_func %F_fail(%arg0) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %false : i1 to !smt.bv<1>
// CHECK-NEXT:       %6 = smt.apply_func %F_fail(%5) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %true : i1 to !smt.bv<1>
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %7 : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %9 = smt.eq %8, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %10 = smt.and %4, %9
// CHECK-NEXT:       %11 = smt.implies %10, %6
// CHECK-NEXT:       smt.yield %11 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %3
// CHECK-NEXT:   }
// CHECK-NEXT: }


fsm.machine @priority_test() -> (i1) attributes {initialState = "start"} {
  %false = hw.constant 0 : i1
  %true = hw.constant 1 : i1 
  
  fsm.state @start output {
    fsm.output %false : i1
  } transitions {
    fsm.transition @start guard {
      fsm.return %true
    }
    fsm.transition @fail guard {
      fsm.return %true
    }
  }

  fsm.state @fail output {
    fsm.output %false : i1
  } transitions {
    fsm.transition @fail guard {
      fsm.return %true
    }
  }
}
