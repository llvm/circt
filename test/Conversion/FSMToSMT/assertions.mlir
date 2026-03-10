// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     [[C0_I8:%.+]] = hw.constant 0 : i8
// CHECK-NEXT:     [[C50_I8:%.+]] = hw.constant 50 : i8
// CHECK-NEXT:     [[C50_I32:%.+]] = hw.constant 50 : i32
// CHECK-NEXT:     [[C1_i32:%.+]] = hw.constant 1 : i32
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
// CHECK-NEXT:       [[TRUE_0:%.+]] = smt.constant true
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
// CHECK-NEXT:     ^bb0([[ARG7:%.+]]: !smt.bv<8>, [[ARG8:%.+]]: !smt.bv<8>, [[ARG9:%.+]]: !smt.bv<8>, [[ARG10:%.+]]: !smt.bv<32>):
// CHECK-NEXT:       [[FUN3:%.+]] = smt.apply_func [[F_S1]]([[ARG9]], [[ARG10]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[CAST12:%.+]] = builtin.unrealized_conversion_cast [[ARG7]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST13:%.+]] = builtin.unrealized_conversion_cast [[ARG10]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[ADD2:%.+]] = comb.add [[CAST13]], [[C1_i32]] : i32
// CHECK-NEXT:       [[CMP2:%.+]] = comb.icmp ult [[ADD2]], [[C50_I32]] : i32
// CHECK-NEXT:       [[CAST14:%.+]] = builtin.unrealized_conversion_cast [[ADD2]] : i32 to !smt.bv<32>
// CHECK-NEXT:       [[CAST15:%.+]] = builtin.unrealized_conversion_cast [[ARG8]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST16:%.+]] = builtin.unrealized_conversion_cast [[CAST14]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[CAST17:%.+]] = builtin.unrealized_conversion_cast [[CAST15]] : i8 to !smt.bv<8>
// CHECK-NEXT:       [[FUN4:%.+]] = smt.apply_func [[F_S0]]([[CAST17]], [[CAST14]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[CAST18:%.+]] = builtin.unrealized_conversion_cast [[ARG7]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST19:%.+]] = builtin.unrealized_conversion_cast [[ARG10]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[TRUE_1:%.+]] = smt.constant true
// CHECK-NEXT:       [[AND2:%.+]] = smt.and [[FUN3]], [[TRUE_1]]
// CHECK-NEXT:       [[IMP2:%.+]] = smt.implies [[AND2]], [[FUN4]]
// CHECK-NEXT:       smt.yield [[IMP2]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL2]]
// CHECK-NEXT:     [[FORALL3:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG11:%.+]]: !smt.bv<8>, [[ARG12:%.+]]: !smt.bv<8>, [[ARG13:%.+]]: !smt.bv<32>):
// CHECK-NEXT:       [[CAST20:%.+]] = builtin.unrealized_conversion_cast [[ARG11]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST21:%.+]] = builtin.unrealized_conversion_cast [[ARG13]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[TRUE_2:%.+]] = smt.constant true
// CHECK-NEXT:       [[CAST22:%.+]] = builtin.unrealized_conversion_cast [[TRUE]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST23:%.+]] = builtin.unrealized_conversion_cast [[CAST22]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C3:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ3:%.+]] = smt.eq [[CAST23]], [[C3]] : !smt.bv<1>
// CHECK-NEXT:       [[FUN5:%.+]] = smt.apply_func [[F_S0]]([[ARG12]], [[ARG13]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[IMP3:%.+]] = smt.implies [[FUN5]], [[EQ3]]
// CHECK-NEXT:       smt.yield [[IMP3]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL3]]
// CHECK-NEXT:     [[FORALL4:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG14:%.+]]: !smt.bv<8>, [[ARG15:%.+]]: !smt.bv<8>, [[ARG16:%.+]]: !smt.bv<32>):
// CHECK-NEXT:       [[CAST24:%.+]] = builtin.unrealized_conversion_cast [[ARG14]] : !smt.bv<8> to i8
// CHECK-NEXT:       [[CAST25:%.+]] = builtin.unrealized_conversion_cast [[ARG16]] : !smt.bv<32> to i32
// CHECK-NEXT:       [[TRUE_3:%.+]] = smt.constant true
// CHECK-NEXT:       [[CAST26:%.+]] = builtin.unrealized_conversion_cast [[TRUE]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST27:%.+]] = builtin.unrealized_conversion_cast [[CAST26]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C4:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ4:%.+]] = smt.eq [[CAST27]], [[C4]] : !smt.bv<1>
// CHECK-NEXT:       [[FUN6:%.+]] = smt.apply_func [[F_S0]]([[ARG15]], [[ARG16]]) : !smt.func<(!smt.bv<8>, !smt.bv<32>) !smt.bool>
// CHECK-NEXT:       [[IMP4:%.+]] = smt.implies [[FUN6]], [[EQ4]]
// CHECK-NEXT:       smt.yield [[IMP4]] : !smt.bool
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
