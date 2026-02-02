// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     [[C0_I8:%.+]] = hw.constant 0 : i8
// CHECK-NEXT:     [[C50_I8:%.+]] = hw.constant 50 : i8
// CHECK-NEXT:     [[C50_I32:%.+]] = hw.constant 50 : i32
// CHECK-NEXT:     [[C1_I32:%.+]] = hw.constant 1 : i32
// CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
// CHECK-NEXT:     [[F_S0:%.+]] = smt.declare_fun "F_S0" : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:     [[F_S1:%.+]] = smt.declare_fun "F_S1" : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:     [[FORALL0:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG0:%.+]]: !smt.bv<8>, [[ARG1:%.+]]: !smt.bv<8>, [[ARG2:%.+]]: !smt.bv<32>):
// CHECK-NEXT:       [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[ARG0]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[ARG2]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[CAST0]] : i8 to !smt.bv<8>
// CHECK-NEXT:       [[C0_BV32:%.+]] = smt.bv.constant #smt.bv<0> : !smt.bv<32>
// CHECK-NEXT:       [[FUN0:%.+]] = smt.apply_func [[F_S0]]([[CAST2]], [[C0_BV32]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       smt.yield [[FUN0]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL0]]
// CHECK-NEXT:     [[FORALL1:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG0_1:%.+]]: !smt.bv<8>, [[ARG1_1:%.+]]: !smt.bv<8>, [[ARG2_1:%.+]]: !smt.bv<8>, [[ARG3_1:%.+]]: !smt.bv<32>):
// CHECK-NEXT:       [[FUN1:%.+]] = smt.apply_func [[F_S0]]([[ARG2_1]], [[ARG3_1]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[CAST3:%.+]] = builtin.unrealized_conversion_cast [[ARG0_1]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST4:%.+]] = builtin.unrealized_conversion_cast [[ARG3_1]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[CAST5:%.+]] = builtin.unrealized_conversion_cast [[ARG1_1]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST6:%.+]] = builtin.unrealized_conversion_cast [[ARG3_1]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[CAST7:%.+]] = builtin.unrealized_conversion_cast [[C0_I8]] : i8 to !smt.bv<8>
// CHECK-NEXT:       [[FUN2:%.+]] = smt.apply_func [[F_S1]]([[CAST7]], [[ARG3_1]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[CAST8:%.+]] = builtin.unrealized_conversion_cast [[ARG0_1]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST9:%.+]] = builtin.unrealized_conversion_cast [[ARG3_1]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[ICMP0:%.+]] = comb.icmp ult [[CAST8]], [[C50_I8]] : i8
// CHECK-NEXT:       [[ICMP1:%.+]] = comb.icmp ugt [[CAST8]], [[C0_I8]] : i8
// CHECK-NEXT:       [[CAST10:%.+]] = builtin.unrealized_conversion_cast [[ICMP1]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST11:%.+]] = builtin.unrealized_conversion_cast [[CAST10]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C_1_BV1:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ0:%.+]] = smt.eq [[CAST11]], [[C_1_BV1]] : !smt.bv<1>
// CHECK-NEXT:       [[AND0:%.+]] = smt.and [[FUN1]], [[EQ0]]
// CHECK-NEXT:       [[IMPLIES0:%.+]] = smt.implies [[AND0]], [[FUN2]]
// CHECK-NEXT:       smt.yield [[IMPLIES0]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL1]]
// CHECK-NEXT:     [[FORALL2:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG0_2:%.+]]: !smt.bv<8>, [[ARG1_2:%.+]]: !smt.bv<8>, [[ARG2_2:%.+]]: !smt.bv<8>, [[ARG3_2:%.+]]: !smt.bv<32>):
// CHECK-NEXT:       [[FUN3:%.+]] = smt.apply_func [[F_S1]]([[ARG2_2]], [[ARG3_2]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[CAST12:%.+]] = builtin.unrealized_conversion_cast [[ARG0_2]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST13:%.+]] = builtin.unrealized_conversion_cast [[ARG3_2]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[ADD0:%.+]] = comb.add [[CAST13]], [[C1_I32]] : i32
// CHECK-NEXT:       [[ICMP2:%.+]] = comb.icmp ult [[ADD0]], [[C50_I32]] : i32
// CHECK-NEXT:       [[CAST14:%.+]] = builtin.unrealized_conversion_cast [[ADD0]] : i32 to !smt.bv<32>
// CHECK-NEXT:       [[CAST15:%.+]] = builtin.unrealized_conversion_cast [[ARG1_2]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST16:%.+]] = builtin.unrealized_conversion_cast [[CAST14]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[CAST17:%.+]] = builtin.unrealized_conversion_cast [[CAST15]] : i8 to !smt.bv<8>
// CHECK-NEXT:       [[FUN4:%.+]] = smt.apply_func [[F_S0]]([[CAST17]], [[CAST14]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[IMPLIES1:%.+]] = smt.implies [[FUN3]], [[FUN4]]
// CHECK-NEXT:       smt.yield [[IMPLIES1]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL2]]
// CHECK-NEXT:     [[FORALL3:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG0_3:%.+]]: !smt.bv<8>, [[ARG1_3:%.+]]: !smt.bv<8>, [[ARG2_3:%.+]]: !smt.bv<32>):
// CHECK-NEXT:       [[CAST18:%.+]] = builtin.unrealized_conversion_cast [[ARG0_3]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST19:%.+]] = builtin.unrealized_conversion_cast [[ARG2_3]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[TRUE_0:%.+]] = smt.constant true
// CHECK-NEXT:       [[CAST20:%.+]] = builtin.unrealized_conversion_cast [[TRUE]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST21:%.+]] = builtin.unrealized_conversion_cast [[CAST20]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C_1_BV1_0:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ1:%.+]] = smt.eq [[CAST21]], [[C_1_BV1_0]] : !smt.bv<1>
// CHECK-NEXT:       [[FUN5:%.+]] = smt.apply_func [[F_S0]]([[ARG1_3]], [[ARG2_3]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[IMPLIES2:%.+]] = smt.implies [[FUN5]], [[EQ1]]
// CHECK-NEXT:       smt.yield [[IMPLIES2]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL3]]
// CHECK-NEXT:     [[FORALL4:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG0_4:%.+]]: !smt.bv<8>, [[ARG1_4:%.+]]: !smt.bv<8>, [[ARG2_4:%.+]]: !smt.bv<32>):
// CHECK-NEXT:       [[CAST22:%.+]] = builtin.unrealized_conversion_cast [[ARG0_4]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST23:%.+]] = builtin.unrealized_conversion_cast [[ARG2_4]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[TRUE_1:%.+]] = smt.constant true
// CHECK-NEXT:       [[CAST24:%.+]] = builtin.unrealized_conversion_cast [[TRUE]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST25:%.+]] = builtin.unrealized_conversion_cast [[CAST24]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C_1_BV1_1:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ2:%.+]] = smt.eq [[CAST25]], [[C_1_BV1_1]] : !smt.bv<1>
// CHECK-NEXT:       [[FUN6:%.+]] = smt.apply_func [[F_S0]]([[ARG1_4]], [[ARG2_4]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[IMPLIES3:%.+]] = smt.implies [[FUN6]], [[EQ2]]
// CHECK-NEXT:       smt.yield [[IMPLIES3]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL4]]
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
