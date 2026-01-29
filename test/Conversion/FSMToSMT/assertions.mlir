// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     %c0_i8 = hw.constant 0 : i8
// CHECK-NEXT:     %c50_i8 = hw.constant 50 : i8
// CHECK-NEXT:     %c50_i32 = hw.constant 50 : i32
// CHECK-NEXT:     %c1_i32 = hw.constant 1 : i32
// CHECK-NEXT:     %true = hw.constant true
// CHECK-NEXT:     %F_S0 = smt.declare_fun "F_S0" : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:     %F_S1 = smt.declare_fun "F_S1" : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:     %0 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>, %arg2: !smt.bv<32>):
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<8> to i8
// CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<32> to i32
// CHECK-NEXT:       %c0_i32 = hw.constant 0 : i32
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %5 : i8 to !smt.bv<8>
// CHECK-NEXT:       %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
// CHECK-NEXT:       %8 = smt.apply_func %F_S0(%7, %c0_bv32) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       smt.yield %8 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %0
// CHECK-NEXT:     %1 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>, %arg2: !smt.bv<8>, %arg3: !smt.bv<32>):
// CHECK-NEXT:       %5 = smt.apply_func %F_S0(%arg2, %arg3) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<8> to i8
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %arg3 : !smt.bv<32> to i32
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %arg1 : !smt.bv<8> to i8
// CHECK-NEXT:       %9 = builtin.unrealized_conversion_cast %arg3 : !smt.bv<32> to i32
// CHECK-NEXT:       %10 = builtin.unrealized_conversion_cast %c0_i8 : i8 to !smt.bv<8>
// CHECK-NEXT:       %11 = smt.apply_func %F_S1(%10, %arg3) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       %12 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<8> to i8
// CHECK-NEXT:       %13 = builtin.unrealized_conversion_cast %arg3 : !smt.bv<32> to i32
// CHECK-NEXT:       %14 = comb.icmp ult %12, %c50_i8 : i8
// CHECK-NEXT:       %15 = comb.icmp ugt %12, %c0_i8 : i8
// CHECK-NEXT:       %16 = builtin.unrealized_conversion_cast %15 : i1 to !smt.bv<1>
// CHECK-NEXT:       %17 = builtin.unrealized_conversion_cast %16 : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %18 = smt.eq %17, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %19 = smt.and %5, %18
// CHECK-NEXT:       %20 = smt.implies %19, %11
// CHECK-NEXT:       smt.yield %20 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %1
// CHECK-NEXT:     %2 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>, %arg2: !smt.bv<8>, %arg3: !smt.bv<32>):
// CHECK-NEXT:       %5 = smt.apply_func %F_S1(%arg2, %arg3) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<8> to i8
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %arg3 : !smt.bv<32> to i32
// CHECK-NEXT:       %8 = comb.add %7, %c1_i32 : i32
// CHECK-NEXT:       %9 = comb.icmp ult %8, %c50_i32 : i32
// CHECK-NEXT:       %10 = builtin.unrealized_conversion_cast %8 : i32 to !smt.bv<32>
// CHECK-NEXT:       %11 = builtin.unrealized_conversion_cast %arg1 : !smt.bv<8> to i8
// CHECK-NEXT:       %12 = builtin.unrealized_conversion_cast %10 : !smt.bv<32> to i32
// CHECK-NEXT:       %13 = builtin.unrealized_conversion_cast %11 : i8 to !smt.bv<8>
// CHECK-NEXT:       %14 = smt.apply_func %F_S0(%13, %10) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       %15 = smt.implies %5, %14
// CHECK-NEXT:       smt.yield %15 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %2
// CHECK-NEXT:     %3 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>, %arg2: !smt.bv<32>):
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<8> to i8
// CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<32> to i32
// CHECK-NEXT:       %true_0 = smt.constant true
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %true : i1 to !smt.bv<1>
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %7 : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %9 = smt.eq %8, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %10 = smt.apply_func %F_S0(%arg1, %arg2) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       %11 = smt.implies %10, %9
// CHECK-NEXT:       smt.yield %11 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %3
// CHECK-NEXT:     %4 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>, %arg2: !smt.bv<32>):
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<8> to i8
// CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<32> to i32
// CHECK-NEXT:       %true_0 = smt.constant true
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %true : i1 to !smt.bv<1>
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %7 : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %9 = smt.eq %8, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %10 = smt.apply_func %F_S0(%arg1, %arg2) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       %11 = smt.implies %10, %9
// CHECK-NEXT:       smt.yield %11 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %4
// CHECK-NEXT:   }
// CHECK-NEXT: }

fsm.machine @guard_assert(%arg0: i8) -> (i8) attributes {initialState = "S0"} {
  %c0_i8 = hw.constant 0 : i8
  %c50_i8 = hw.constant 50 : i8 
  %c50_i32 = hw.constant 50 : i32
  %c1_i32 = hw.constant 1 : i32
  %counter = fsm.variable "counter" {initValue = 0 : i32} : i32
  
  %true = hw.constant true
  
  fsm.state @S0 output {
    verif.assert %true : i1
    fsm.output %arg0 : i8
  } transitions {
    fsm.transition @S1 guard {
      %valid = comb.icmp ult %arg0, %c50_i8 : i8
      %cond = comb.icmp ugt %arg0, %c0_i8 : i8
      fsm.return %cond
    }
  }
  
  fsm.state @S1 output {
    fsm.output %c0_i8 : i8
  } transitions {
    fsm.transition @S0 action {
      %new = comb.add %counter, %c1_i32 : i32
      %valid = comb.icmp ult %new, %c50_i32 : i32
      fsm.update %counter, %new : i32
    }
  }
}
