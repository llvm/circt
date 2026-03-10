// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     [[TRUE:%.+]] = hw.constant true
// CHECK-NEXT:     [[FALSE:%.+]] = hw.constant false
// CHECK-NEXT:     [[F_S0:%.+]] = smt.declare_fun "F_s0" : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:     [[F_S1:%.+]] = smt.declare_fun "F_s1" : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:     [[FORALL0:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG0:%.+]]: !smt.bv<1>):
// CHECK-NEXT:       [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[FALSE]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[APP0:%.+]] = smt.apply_func [[F_S0]]([[CAST0]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       smt.yield [[APP0]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL0]]
// CHECK-NEXT:     [[FORALL1:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG1:%.+]]: !smt.bv<1>):
// CHECK-NEXT:       [[APP1:%.+]] = smt.apply_func [[F_S0]]([[ARG1]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[FALSE]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[APP2:%.+]] = smt.apply_func [[F_S1]]([[CAST1]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[TRUE_0:%.+]] = smt.constant true
// CHECK-NEXT:       [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[TRUE]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST3:%.+]] = builtin.unrealized_conversion_cast [[CAST2]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C1:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ1:%.+]] = smt.eq [[CAST3]], [[C1]] : !smt.bv<1>
// CHECK-NEXT:       [[AND1:%.+]] = smt.and [[APP1]], [[EQ1]]
// CHECK-NEXT:       [[IMP1:%.+]] = smt.implies [[AND1]], [[APP2]]
// CHECK-NEXT:       smt.yield [[IMP1]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL1]]
// CHECK-NEXT:     [[FORALL2:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG2:%.+]]: !smt.bv<1>):
// CHECK-NEXT:       [[APP3:%.+]] = smt.apply_func [[F_S0]]([[ARG2]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[CAST4:%.+]] = builtin.unrealized_conversion_cast [[FALSE]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[APP4:%.+]] = smt.apply_func [[F_S0]]([[CAST4]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[TRUE_1:%.+]] = smt.constant true
// CHECK-NEXT:       [[CAST5:%.+]] = builtin.unrealized_conversion_cast [[TRUE]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST6:%.+]] = builtin.unrealized_conversion_cast [[CAST5]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C2:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ2:%.+]] = smt.eq [[CAST6]], [[C2]] : !smt.bv<1>
// CHECK-NEXT:       [[NOT2:%.+]] = smt.not [[EQ2]]
// CHECK-NEXT:       [[AND2:%.+]] = smt.and [[TRUE_1]], [[NOT2]]
// CHECK-NEXT:       [[AND2_0:%.+]] = smt.and [[APP3]], [[AND2]]
// CHECK-NEXT:       [[IMP2:%.+]] = smt.implies [[AND2_0]], [[APP4]]
// CHECK-NEXT:       smt.yield [[IMP2]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL2]]
// CHECK-NEXT:   }
// CHECK-NEXT: }

fsm.machine @fsm2() -> (i1) attributes {initialState = "s0"} {
    %t = hw.constant 1 : i1
    %f = hw.constant 0 : i1
    
    fsm.state @s0 output {
        fsm.output %f : i1
    } transitions {
        fsm.transition @s1 guard {   
        fsm.return %t
        }
        fsm.transition @s0           
    }
    fsm.state @s1 output { 
        fsm.output %f : i1 
    } transitions { 
    }

}