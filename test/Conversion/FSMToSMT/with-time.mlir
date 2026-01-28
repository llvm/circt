// RUN: circt-opt --convert-fsm-to-smt="with-time=true" %s | FileCheck %s

// CHECK: module @fsm_with_time {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     %c0_i32 = hw.constant 0 : i32
// CHECK-NEXT:     %c1_i32 = hw.constant 1 : i32
// CHECK-NEXT:     %F_S0 = smt.declare_fun "F_S0" : !smt.func<(!smt.bv<8>, !smt.bv<32>, !smt.bv<5>) !smt.bool>
// CHECK-NEXT:     %F_S1 = smt.declare_fun "F_S1" : !smt.func<(!smt.bv<8>, !smt.bv<32>, !smt.bv<5>) !smt.bool>
// CHECK-NEXT:     %0 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>, %arg2: !smt.bv<32>, %arg3: !smt.bv<5>):
// CHECK-NEXT:       %3 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<8> to i8
// CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %arg2 : !smt.bv<32> to i32
// CHECK-NEXT:       %c1_i32_0 = hw.constant 1 : i32
// CHECK-NEXT:       %5 = comb.extract %c1_i32_0 from 0 : (i32) -> i8
// CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %5 : i8 to !smt.bv<8>
// CHECK-NEXT:       %c1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
// CHECK-NEXT:       %c0_bv5 = smt.bv.constant #smt.bv<0> : !smt.bv<5>
// CHECK-NEXT:       %7 = smt.apply_func %F_S0(%6, %c1_bv32, %c0_bv5) : !smt.func<(!smt.bv<8>, !smt.bv<32>, !smt.bv<5>) !smt.bool>
// CHECK-NEXT:       smt.yield %7 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %0
// CHECK-NEXT:     %1 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>, %arg2: !smt.bv<8>, %arg3: !smt.bv<32>, %arg4: !smt.bv<5>):
// CHECK-NEXT:       %3 = smt.apply_func %F_S0(%arg2, %arg3, %arg4) : !smt.func<(!smt.bv<8>, !smt.bv<32>, !smt.bv<5>) !smt.bool>
// CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<8> to i8
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %arg3 : !smt.bv<32> to i32
// CHECK-NEXT:       %6 = comb.add %5, %c1_i32 : i32
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %6 : i32 to !smt.bv<32>
// CHECK-NEXT:       %8 = builtin.unrealized_conversion_cast %arg1 : !smt.bv<8> to i8
// CHECK-NEXT:       %9 = builtin.unrealized_conversion_cast %7 : !smt.bv<32> to i32
// CHECK-NEXT:       %10 = builtin.unrealized_conversion_cast %8 : i8 to !smt.bv<8>
// CHECK-NEXT:       %c1_bv5 = smt.bv.constant #smt.bv<1> : !smt.bv<5>
// CHECK-NEXT:       %11 = smt.bv.add %arg4, %c1_bv5 : !smt.bv<5>
// CHECK-NEXT:       %12 = smt.apply_func %F_S1(%10, %7, %11) : !smt.func<(!smt.bv<8>, !smt.bv<32>, !smt.bv<5>) !smt.bool>
// CHECK-NEXT:       %13 = smt.implies %3, %12
// CHECK-NEXT:       smt.yield %13 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %1
// CHECK-NEXT:     %2 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>, %arg2: !smt.bv<8>, %arg3: !smt.bv<32>, %arg4: !smt.bv<5>):
// CHECK-NEXT:       %3 = smt.apply_func %F_S1(%arg2, %arg3, %arg4) : !smt.func<(!smt.bv<8>, !smt.bv<32>, !smt.bv<5>) !smt.bool>
// CHECK-NEXT:       %4 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<8> to i8
// CHECK-NEXT:       %5 = builtin.unrealized_conversion_cast %arg3 : !smt.bv<32> to i32
// CHECK-NEXT:       %6 = builtin.unrealized_conversion_cast %arg1 : !smt.bv<8> to i8
// CHECK-NEXT:       %7 = builtin.unrealized_conversion_cast %arg3 : !smt.bv<32> to i32
// CHECK-NEXT:       %8 = comb.extract %7 from 0 : (i32) -> i8
// CHECK-NEXT:       %9 = builtin.unrealized_conversion_cast %8 : i8 to !smt.bv<8>
// CHECK-NEXT:       %c1_bv5 = smt.bv.constant #smt.bv<1> : !smt.bv<5>
// CHECK-NEXT:       %10 = smt.bv.add %arg4, %c1_bv5 : !smt.bv<5>
// CHECK-NEXT:       %11 = smt.apply_func %F_S0(%9, %arg3, %10) : !smt.func<(!smt.bv<8>, !smt.bv<32>, !smt.bv<5>) !smt.bool>
// CHECK-NEXT:       %12 = smt.implies %3, %11
// CHECK-NEXT:       smt.yield %12 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %2
// CHECK-NEXT:   }
// CHECK-NEXT: }


module @fsm_with_time {
  
  fsm.machine @timed(%arg0: i8) -> (i8) attributes {initialState = "S0"} {
    %c0_i32 = hw.constant 0 : i32
    %c1_i32 = hw.constant 1 : i32
    %counter = fsm.variable "counter" {initValue = 1 : i32} : i32
    
    fsm.state @S0 output {
      %0 = comb.extract %counter from 0 : (i32) -> i8
      fsm.output %0 : i8
    } transitions {
      fsm.transition @S1 action {
        %new = comb.add %counter, %c1_i32 : i32
        fsm.update %counter, %new : i32
      }
    }
    
    fsm.state @S1 output {
      fsm.output %arg0 : i8
    } transitions {
      fsm.transition @S0
    }
  }
}