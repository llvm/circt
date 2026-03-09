// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s


// CHECK: module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     [[F_START:%.+]] = smt.declare_fun "F_start" : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:     [[F_FAIL:%.+]] = smt.declare_fun "F_fail" : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:     [[FORALL0:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG0:%.+]]: !smt.bv<1>):
// CHECK-NEXT:       [[FALSE0:%.+]] = hw.constant false
// CHECK-NEXT:       [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[FALSE0]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[APP0:%.+]] = smt.apply_func [[F_START]]([[CAST0]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       smt.yield [[APP0]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL0]]
// CHECK-NEXT:     [[FORALL1:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG1:%.+]]: !smt.bv<1>):
// CHECK-NEXT:       [[APP1:%.+]] = smt.apply_func [[F_START]]([[ARG1]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[FALSE1:%.+]] = hw.constant false
// CHECK-NEXT:       [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[FALSE1]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[APP2:%.+]] = smt.apply_func [[F_START]]([[CAST1]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[TRUE1:%.+]] = hw.constant true
// CHECK-NEXT:       [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[TRUE1]] : i1 to !smt.bv<1>
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
// CHECK-NEXT:       [[APP3:%.+]] = smt.apply_func [[F_START]]([[ARG2]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[FALSE2:%.+]] = hw.constant false
// CHECK-NEXT:       [[CAST4:%.+]] = builtin.unrealized_conversion_cast [[FALSE2]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[APP4:%.+]] = smt.apply_func [[F_FAIL]]([[CAST4]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[TRUE2:%.+]] = hw.constant true
// CHECK-NEXT:       [[CAST5:%.+]] = builtin.unrealized_conversion_cast [[TRUE2]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST6:%.+]] = builtin.unrealized_conversion_cast [[CAST5]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C2:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ2:%.+]] = smt.eq [[CAST6]], [[C2]] : !smt.bv<1>
// CHECK-NEXT:       [[TRUE2_0:%.+]] = hw.constant true
// CHECK-NEXT:       [[CAST7:%.+]] = builtin.unrealized_conversion_cast [[TRUE2_0]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST8:%.+]] = builtin.unrealized_conversion_cast [[CAST7]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C2_1:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ2_1:%.+]] = smt.eq [[CAST8]], [[C2_1]] : !smt.bv<1>
// CHECK-NEXT:       [[NOT2:%.+]] = smt.not [[EQ2_1]]
// CHECK-NEXT:       [[AND2:%.+]] = smt.and [[EQ2]], [[NOT2]]
// CHECK-NEXT:       [[AND2_1:%.+]] = smt.and [[APP3]], [[AND2]]
// CHECK-NEXT:       [[IMP2:%.+]] = smt.implies [[AND2_1]], [[APP4]]
// CHECK-NEXT:       smt.yield [[IMP2]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL2]]
// CHECK-NEXT:     [[FORALL3:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG3:%.+]]: !smt.bv<1>):
// CHECK-NEXT:       [[APP5:%.+]] = smt.apply_func [[F_FAIL]]([[ARG3]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[FALSE3:%.+]] = hw.constant false
// CHECK-NEXT:       [[CAST9:%.+]] = builtin.unrealized_conversion_cast [[FALSE3]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[APP6:%.+]] = smt.apply_func [[F_FAIL]]([[CAST9]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[TRUE3:%.+]] = hw.constant true
// CHECK-NEXT:       [[CAST10:%.+]] = builtin.unrealized_conversion_cast [[TRUE3]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST11:%.+]] = builtin.unrealized_conversion_cast [[CAST10]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C3:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ3:%.+]] = smt.eq [[CAST11]], [[C3]] : !smt.bv<1>
// CHECK-NEXT:       [[AND3:%.+]] = smt.and [[APP5]], [[EQ3]]
// CHECK-NEXT:       [[IMP3:%.+]] = smt.implies [[AND3]], [[APP6]]
// CHECK-NEXT:       smt.yield [[IMP3]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL3]]
// CHECK-NEXT:     [[FORALL4:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG4:%.+]]: !smt.bv<1>):
// CHECK-NEXT:       [[TRUE4:%.+]] = smt.constant true
// CHECK-NEXT:       [[FALSE4:%.+]] = hw.constant false
// CHECK-NEXT:       [[CAST12:%.+]] = builtin.unrealized_conversion_cast [[FALSE4]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST13:%.+]] = builtin.unrealized_conversion_cast [[CAST12]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C4:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ4:%.+]] = smt.eq [[CAST13]], [[C4]] : !smt.bv<1>
// CHECK-NEXT:       [[APP7:%.+]] = smt.apply_func [[F_FAIL]]([[ARG4]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[IMP4:%.+]] = smt.implies [[APP7]], [[EQ4]]
// CHECK-NEXT:       smt.yield [[IMP4]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL4]]
// CHECK-NEXT:     [[FORALL5:%.+]] = smt.forall {
// CHECK-NEXT:     ^bb0([[ARG5:%.+]]: !smt.bv<1>):
// CHECK-NEXT:       [[TRUE5:%.+]] = smt.constant true
// CHECK-NEXT:       [[FALSE5:%.+]] = hw.constant false
// CHECK-NEXT:       [[CAST14:%.+]] = builtin.unrealized_conversion_cast [[FALSE5]] : i1 to !smt.bv<1>
// CHECK-NEXT:       [[CAST15:%.+]] = builtin.unrealized_conversion_cast [[CAST14]] : !smt.bv<1> to !smt.bv<1>
// CHECK-NEXT:       [[C5:%.+]] = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       [[EQ5:%.+]] = smt.eq [[CAST15]], [[C5]] : !smt.bv<1>
// CHECK-NEXT:       [[APP8:%.+]] = smt.apply_func [[F_FAIL]]([[ARG5]]) : !smt.func<(!smt.bv<1>) !smt.bool>
// CHECK-NEXT:       [[IMP5:%.+]] = smt.implies [[APP8]], [[EQ5]]
// CHECK-NEXT:       smt.yield [[IMP5]] : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert [[FORALL5]]
// CHECK-NEXT:   }
// CHECK-NEXT: }


fsm.machine @priority_test() -> (i1) attributes {initialState = "start"} {

  fsm.state @start output {
    %false = hw.constant 0 : i1
    fsm.output %false : i1
  } transitions {
    fsm.transition @start guard {
      %true = hw.constant 1 : i1 
      fsm.return %true
    }
    fsm.transition @fail guard {
      %true = hw.constant 1 : i1
      fsm.return %true
    }
  }

  fsm.state @fail output {
    %f = hw.constant 0 : i1
    verif.assert %f : i1
    fsm.output %f : i1
  } transitions {
    fsm.transition @fail guard {
      %true = hw.constant 1 : i1
      fsm.return %true
    }
  }
  }
