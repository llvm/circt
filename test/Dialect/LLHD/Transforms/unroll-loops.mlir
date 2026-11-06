// RUN: circt-opt --llhd-unroll-loops %s | FileCheck %s

func.func private @marker()

// CHECK-LABEL: @SimpleLoop
hw.module @SimpleLoop(out x : i42) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  %c42_i42 = hw.constant 42 : i42
  // Loop of the form:
  //   x = 0
  //   for (i = 0; i < 3; ++i)
  //     x += 42
  // CHECK: llhd.combinational
  %0 = llhd.combinational -> i42 {
    // CHECK-NEXT:   cf.br [[ENTRY:\^.+]](%c0_i42 : i42)
    cf.br ^header(%c0_i42, %c0_i42 : i42, i42)
  ^header(%i: i42, %x: i42):  // 2 preds: ^bb0, ^body
    %1 = comb.icmp slt %i, %c3_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:  // pred: ^header
    // CHECK-NEXT: [[ENTRY]]([[X0:%.+]]: i42):
    // CHECK-NEXT:   [[X1:%.+]] = comb.add [[X0]], %c42_i42
    // CHECK-NEXT:   [[X2:%.+]] = comb.add [[X1]], %c42_i42
    // CHECK-NEXT:   [[X3:%.+]] = comb.add [[X2]], %c42_i42
    // CHECK-NEXT:   cf.br [[EXIT:\^.+]]
    %2 = comb.add %x, %c42_i42 : i42
    %ip = comb.add %i, %c1_i42 : i42
    cf.br ^header(%ip, %2 : i42, i42)
  ^exit:  // pred: ^header
    // CHECK-NEXT: [[EXIT]]:
    // CHECK-NEXT:   llhd.yield [[X3]]
    llhd.yield %x : i42
  }
  hw.output %0 : i42
}

// CHECK-LABEL: @SimpleDescendingLoop
hw.module @SimpleDescendingLoop(out x : i42) {
  %c0_i42 = hw.constant 0 : i42
  %c_min1_i42 = hw.constant -1 : i42
  %c3_i42 = hw.constant 3 : i42
  %c42_i42 = hw.constant 42 : i42
  // Loop of the form:
  //   x = 0
  //   for (i = 3; i >= 0; i--)
  //     x += 42
  // CHECK: llhd.combinational
  %0 = llhd.combinational -> i42 {
    // CHECK-NEXT:   cf.br [[ENTRY:\^.+]](%c0_i42 : i42)
    cf.br ^header(%c3_i42, %c0_i42 : i42, i42)
  ^header(%i: i42, %x: i42):  // 2 preds: ^bb0, ^body
    %1 = comb.icmp sgt %i, %c_min1_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:  // pred: ^header
    // CHECK-NEXT: [[ENTRY]]([[X0:%.+]]: i42):
    // CHECK-NEXT:   [[X1:%.+]] = comb.add [[X0]], %c42_i42
    // CHECK-NEXT:   [[X2:%.+]] = comb.add [[X1]], %c42_i42
    // CHECK-NEXT:   [[X3:%.+]] = comb.add [[X2]], %c42_i42
    // CHECK-NEXT:   [[X4:%.+]] = comb.add [[X3]], %c42_i42
    // CHECK-NEXT:   cf.br [[EXIT:\^.+]]
    %2 = comb.add %x, %c42_i42 : i42
    %ip = comb.add %i, %c_min1_i42 : i42
    cf.br ^header(%ip, %2 : i42, i42)
  ^exit:  // pred: ^header
    // CHECK-NEXT: [[EXIT]]:
    // CHECK-NEXT:   llhd.yield [[X4]]
    llhd.yield %x : i42
  }
  hw.output %0 : i42
}

// CHECK-LABEL: @StridedLoopWithOffset
hw.module @StridedLoopWithOffset(out x : i42) {
  %c0_i42 = hw.constant 0 : i42
  %c3_i42 = hw.constant 3 : i42
  %c13_i42 = hw.constant 13 : i42
  // Loop of the form:
  //   x = 0
  //   for (i = 3; i < 13; i += 3)
  //     x += i
  // CHECK: llhd.combinational
  %0 = llhd.combinational -> i42 {
    // CHECK-NEXT:   cf.br [[ENTRY:\^.+]](%c0_i42 : i42)
    cf.br ^header(%c3_i42, %c0_i42 : i42, i42)
  ^header(%i: i42, %x: i42):  // 2 preds: ^bb0, ^body
    %1 = comb.icmp slt %i, %c13_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:  // pred: ^header
    // CHECK-NEXT: [[ENTRY]]([[X0:%.+]]: i42):
    // CHECK-NEXT:   [[C3:%.+]] = hw.constant 3
    // CHECK-NEXT:   [[X1:%.+]] = comb.add [[X0]], [[C3]]
    // CHECK-NEXT:   [[C6:%.+]] = hw.constant 6
    // CHECK-NEXT:   [[X2:%.+]] = comb.add [[X1]], [[C6]]
    // CHECK-NEXT:   [[C9:%.+]] = hw.constant 9
    // CHECK-NEXT:   [[X3:%.+]] = comb.add [[X2]], [[C9]]
    // CHECK-NEXT:   [[C12:%.+]] = hw.constant 12
    // CHECK-NEXT:   [[X4:%.+]] = comb.add [[X3]], [[C12]]
    // CHECK-NEXT:   cf.br [[EXIT:\^.+]]
    %2 = comb.add %x, %i : i42
    %ip = comb.add %i, %c3_i42 : i42
    cf.br ^header(%ip, %2 : i42, i42)
  ^exit:  // pred: ^header
    // CHECK-NEXT: [[EXIT]]:
    // CHECK-NEXT:   llhd.yield [[X4]]
    llhd.yield %x : i42
  }
  hw.output %0 : i42
}

// CHECK-LABEL: @StridedDescendingLoopWithOffset
hw.module @StridedDescendingLoopWithOffset(out x : i42) {
  %c0_i42 = hw.constant 0 : i42
  %c3_i42 = hw.constant 3 : i42
  %c13_i42 = hw.constant 13 : i42
  %c_min3_i42 = hw.constant -3 : i42
  // Loop of the form:
  //   x = 0
  //   for (i = 13; i > 3; i -= 3)
  //     x += i
  // CHECK: llhd.combinational
  %0 = llhd.combinational -> i42 {
    // CHECK-NEXT:   cf.br [[ENTRY:\^.+]](%c0_i42 : i42)
    cf.br ^header(%c13_i42, %c0_i42 : i42, i42)
  ^header(%i: i42, %x: i42):  // 2 preds: ^bb0, ^body
    %1 = comb.icmp sgt %i, %c3_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:  // pred: ^header
    // CHECK-NEXT: [[ENTRY]]([[X0:%.+]]: i42):
    // CHECK-NEXT:   [[C3:%.+]] = hw.constant 13
    // CHECK-NEXT:   [[X1:%.+]] = comb.add [[X0]], [[C3]]
    // CHECK-NEXT:   [[C6:%.+]] = hw.constant 10
    // CHECK-NEXT:   [[X2:%.+]] = comb.add [[X1]], [[C6]]
    // CHECK-NEXT:   [[C9:%.+]] = hw.constant 7
    // CHECK-NEXT:   [[X3:%.+]] = comb.add [[X2]], [[C9]]
    // CHECK-NEXT:   [[C12:%.+]] = hw.constant 4
    // CHECK-NEXT:   [[X4:%.+]] = comb.add [[X3]], [[C12]]
    // CHECK-NEXT:   cf.br [[EXIT:\^.+]]
    %2 = comb.add %x, %i : i42
    %ip = comb.add %i, %c_min3_i42 : i42
    cf.br ^header(%ip, %2 : i42, i42)
  ^exit:  // pred: ^header
    // CHECK-NEXT: [[EXIT]]:
    // CHECK-NEXT:   llhd.yield [[X4]]
    llhd.yield %x : i42
  }
  hw.output %0 : i42
}

// CHECK-LABEL: @TwoNestedLoops
hw.module @TwoNestedLoops(out x : i42) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c2_i42 = hw.constant 2 : i42
  %c3_i42 = hw.constant 3 : i42
  %c42_i42 = hw.constant 42 : i42
  // Loop of the form:
  //   x = 0
  //   for (i = 0; i < 2; ++i)
  //     for (j = 0; j < 3; ++j)
  //       x += 42
  // CHECK: llhd.combinational
  %0 = llhd.combinational -> i42 {
    // CHECK-NEXT:   cf.br [[ENTRY:\^.+]](%c0_i42 : i42)
    cf.br ^outerHeader(%c0_i42, %c0_i42 : i42, i42)
  ^outerHeader(%i: i42, %x1: i42):  // 2 preds: ^bb0, ^innerExit
    %1 = comb.icmp slt %i, %c2_i42 : i42
    cf.cond_br %1, ^innerHeader(%c0_i42, %x1 : i42, i42), ^outerExit
  ^innerHeader(%j: i42, %x2: i42):  // 2 preds: ^outerHeader, ^innerBody
    %2 = comb.icmp slt %j, %c3_i42 : i42
    cf.cond_br %2, ^innerBody, ^innerExit
  ^innerBody:  // pred: ^innerHeader
    // CHECK-NEXT: [[ENTRY]]([[X0:%.+]]: i42):
    // CHECK-NEXT:   [[X1:%.+]] = comb.add [[X0]], %c42_i42
    // CHECK-NEXT:   [[X2:%.+]] = comb.add [[X1]], %c42_i42
    // CHECK-NEXT:   [[X3:%.+]] = comb.add [[X2]], %c42_i42
    // CHECK-NEXT:   [[X4:%.+]] = comb.add [[X3]], %c42_i42
    // CHECK-NEXT:   [[X5:%.+]] = comb.add [[X4]], %c42_i42
    // CHECK-NEXT:   [[X6:%.+]] = comb.add [[X5]], %c42_i42
    // CHECK-NEXT:   cf.br [[EXIT:\^.+]]
    %7 = comb.add %x2, %c42_i42 : i42
    %jp = comb.add %j, %c1_i42 : i42
    cf.br ^innerHeader(%jp, %7 : i42, i42)
  ^innerExit:  // pred: ^innerHeader
    %ip = comb.add %i, %c1_i42 : i42
    cf.br ^outerHeader(%ip, %x2 : i42, i42)
  ^outerExit:  // pred: ^outerHeader
    // CHECK-NEXT: [[EXIT]]:
    // CHECK-NEXT:   llhd.yield [[X6]]
    llhd.yield %x1 : i42
  }
  hw.output %0 : i42
}

// CHECK-LABEL: @SkipLoopWithMultipleBackEdges
hw.module @SkipLoopWithMultipleBackEdges() {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.cond_br
    // CHECK: llhd.yield
    cf.br ^header(%c0_i42 : i42)
  ^header(%0: i42):
    %1 = comb.icmp slt %0, %c3_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:
    %2 = comb.add %0, %c1_i42 : i42
    %3 = comb.extract %2 from 0 : (i42) -> i1
    cf.cond_br %3, ^header(%2 : i42), ^header(%2 : i42)  // two back-edges
  ^exit:
    llhd.yield
  }
}

// A second exit in the body stays in every unrolled copy as a branch out of
// the loop; the header's bound test is the counting exit.
// CHECK-LABEL: @LoopWithSecondExitInBody
hw.module @LoopWithSecondExitInBody() {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK-NEXT: cf.br [[B0:\^.+]]
    // CHECK-NEXT: [[B0]]:
    // CHECK-NEXT: %c0_i42 = hw.constant 0
    // CHECK-NEXT: [[C0:%.+]] = comb.icmp slt %c0_i42, %c3_i42
    // CHECK-NEXT: cf.cond_br [[C0]], [[B1:\^.+]], [[EXIT:\^.+]]
    // CHECK-NEXT: [[B1]]:
    // CHECK-NEXT: %c1_i42 = hw.constant 1
    // CHECK-NEXT: [[C1:%.+]] = comb.icmp slt %c1_i42, %c3_i42
    // CHECK-NEXT: cf.cond_br [[C1]], [[B2:\^.+]], [[EXIT]]
    // CHECK-NEXT: [[B2]]:
    // CHECK-NEXT: %c2_i42 = hw.constant 2
    // CHECK-NEXT: [[C2:%.+]] = comb.icmp slt %c2_i42, %c3_i42
    // CHECK-NEXT: cf.cond_br [[C2]], [[B3:\^.+]], [[EXIT]]
    // CHECK-NEXT: [[B3]]:
    // CHECK-NEXT: cf.br [[EXIT]]
    // CHECK-NEXT: [[EXIT]]:
    // CHECK-NEXT: llhd.yield
    cf.br ^header(%c0_i42 : i42)
  ^header(%0: i42):
    %1 = comb.icmp slt %0, %c3_i42 : i42
    cf.cond_br %1, ^body1, ^exit
  ^body1:
    cf.cond_br %1, ^body2, ^exit
  ^body2:
    %2 = comb.add %0, %c1_i42 : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithUnsupportedExitBranch
hw.module @SkipLoopWithUnsupportedExitBranch() {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.switch
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%c0_i42 : i42)
  ^header(%0: i42):
    cf.switch %0 : i42, [default: ^body, 3: ^exit]
  ^body:
    %2 = comb.add %0, %c1_i42 : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithDynamicLoopBounds
hw.module @SkipLoopWithDynamicLoopBounds(in %a: i42) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%c0_i42 : i42)
  ^header(%0: i42):
    %1 = comb.icmp slt %0, %a : i42
    cf.cond_br %1, ^body, ^exit
  ^body:
    %2 = comb.add %0, %c1_i42 : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithUnsupportedExitCondition
hw.module @SkipLoopWithUnsupportedExitCondition() {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%c0_i42 : i42)
  ^header(%0: i42):
    %1 = comb.extract %0 from 0 : (i42) -> i1
    cf.cond_br %1, ^body, ^exit
  ^body:
    %2 = comb.add %0, %c1_i42 : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithUnsupportedInductionVariable1
hw.module @SkipLoopWithUnsupportedInductionVariable1() {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%c0_i42 : i42)
  ^header(%0: i42):
    %1 = comb.add %0, %0 : i42
    %2 = comb.icmp slt %1, %c3_i42 : i42
    cf.cond_br %2, ^body, ^exit
  ^body:
    %3 = comb.add %0, %c1_i42 : i42
    cf.br ^header(%3 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithUnsupportedInductionVariable2
hw.module @SkipLoopWithUnsupportedInductionVariable2(in %i: i42) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%c0_i42 : i42)
  ^header(%0: i42):
    %1 = comb.icmp slt %0, %c3_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:
    %2 = comb.add %i, %c1_i42 : i42  // <-- uses %i block arg instead of %0
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithMultipleInitialInductionVariableValue
hw.module @SkipLoopWithMultipleInitialInductionVariableValue(in %a: i1) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.cond_br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.cond_br %a, ^header(%c0_i42 : i42), ^header(%c1_i42 : i42)
  ^header(%0: i42):
    %1 = comb.icmp slt %0, %c3_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:
    %2 = comb.add %0, %c1_i42 : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithUnsupportedInitialInductionVariableValue
hw.module @SkipLoopWithUnsupportedInitialInductionVariableValue(in %a: i42) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%a : i42)
  ^header(%0: i42):
    %1 = comb.icmp slt %0, %c3_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:
    %2 = comb.add %0, %c1_i42 : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithDynamicIncrement
hw.module @SkipLoopWithDynamicIncrement(in %a: i42) {
  %c0_i42 = hw.constant 0 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%c0_i42 : i42)
  ^header(%0: i42):
    %1 = comb.icmp slt %0, %c3_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:
    %2 = comb.add %0, %a : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithUnsupportedIncrement
hw.module @SkipLoopWithUnsupportedIncrement() {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%c0_i42 : i42)
  ^header(%0: i42):
    %1 = comb.icmp slt %0, %c3_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:
    %2 = comb.xor %0, %c1_i42 : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithUnsupportedLoopBound
hw.module @SkipLoopWithUnsupportedLoopBound() {
  %c_loop_start = hw.constant  3 : i42
  %c_loop_end   = hw.constant -5000 : i42
  %c_loop_inc   = hw.constant -1 : i42
  // Loop of the form (more than 1024 iterations):
  //   for (i = 3; i > -5000; i--)
  //     func()
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: func.call @marker() : () -> ()
    // CHECK-NOT: func.call @marker() : () -> ()
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%c_loop_start: i42)
  ^header(%i: i42):  // 2 preds: ^bb0, ^bb2
    %1 = comb.icmp sgt %i, %c_loop_end : i42
    cf.cond_br %1, ^body, ^exit
  ^body:  // pred: ^header
    func.call @marker() : () -> ()
    %iInc = comb.add %i, %c_loop_inc : i42
    cf.br ^header(%iInc: i42)
  ^exit:  // pred: ^header
    llhd.yield
  }
  hw.output
}

// CHECK-LABEL: @DontCrashOnSingleBlocks
hw.module @DontCrashOnSingleBlocks() {
  llhd.combinational {
    llhd.yield
  }
}

// CHECK-LABEL: @DegenerateSingleTripLoopWithEq
hw.module @DegenerateSingleTripLoopWithEq() {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK-NEXT: cf.br ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: cf.br ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: llhd.yield
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%1: i32):
    %2 = comb.icmp eq %1, %c0_i32 : i32
    cf.cond_br %2, ^bb2, ^bb3
  ^bb2:
    func.call @marker() : () -> ()
    %4 = comb.add %1, %c1_i32 : i32
    cf.br ^bb1(%4 : i32)
  ^bb3:
    llhd.yield
  }
}

// CHECK-LABEL: @LoopWithUlt
hw.module @LoopWithUlt() {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c3_i32 = hw.constant 3 : i32
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK-NEXT: cf.br ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: cf.br ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: llhd.yield
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%1: i32):
    %2 = comb.icmp ult %1, %c3_i32 : i32
    cf.cond_br %2, ^bb2, ^bb3
  ^bb2:
    func.call @marker() : () -> ()
    %4 = comb.add %1, %c1_i32 : i32
    cf.br ^bb1(%4 : i32)
  ^bb3:
    llhd.yield
  }
}

// CHECK-LABEL: @NestedLoopWithNonConstantInnerBound
hw.module @NestedLoopWithNonConstantInnerBound(in %a: i42, out x : i42) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c2_i42 = hw.constant 2 : i42
  // Loop of the form:
  //   x = 0
  //   for (i = 0; i < 2; ++i)
  //     for (j = i; j < 2; ++j)
  //       x += a
  %0 = llhd.combinational -> i42 {
    // CHECK-NOT: cf.cond_br
    // CHECK: [[X1:%.+]] = comb.add
    // CHECK-NEXT: [[X2:%.+]] = comb.add [[X1]],
    // CHECK-NOT: cf.cond_br
    // CHECK: [[X3:%.+]] = comb.add
    // CHECK-NOT: cf.cond_br
    // CHECK: llhd.yield [[X3]]
    cf.br ^outerHeader(%c0_i42, %c0_i42, %a : i42, i42, i42)
  ^outerHeader(%i: i42, %x1: i42, %a1: i42):
    %1 = comb.icmp ult %i, %c2_i42 : i42
    cf.cond_br %1, ^innerHeader(%i, %x1, %a1 : i42, i42, i42), ^outerExit
  ^innerHeader(%j: i42, %x2: i42, %a2: i42):
    %2 = comb.icmp ult %j, %c2_i42 : i42
    cf.cond_br %2, ^innerBody, ^innerExit
  ^innerBody:
    %xp = comb.add %x2, %a2 : i42
    %jp = comb.add %j, %c1_i42 : i42
    cf.br ^innerHeader(%jp, %xp, %a2 : i42, i42, i42)
  ^innerExit:
    %ip = comb.add %i, %c1_i42 : i42
    cf.br ^outerHeader(%ip, %x2, %a2 : i42, i42, i42)
  ^outerExit:
    llhd.yield %x1 : i42
  }
  hw.output %0 : i42
}

// The trip count is found by stepping the induction variable, so loops that
// count down (`for (i = 2; i >= 0; i--)`, canonicalized to `i > -1` and an add
// of -1), `repeat (3)` (a counter run down to zero, here with a subtract), and
// a `do ... while` that tests the stepped value all unroll. Before this, only
// `for (i = 0; i < N; i++)` did, and the VX backend refused the others.
// CHECK-LABEL: @DownCountingLoop
hw.module @DownCountingLoop() {
  %c-1_i32 = hw.constant -1 : i32
  %c2_i32 = hw.constant 2 : i32
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK-NEXT: cf.br ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: cf.br ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: llhd.yield
    cf.br ^bb1(%c2_i32 : i32)
  ^bb1(%i: i32):
    %0 = comb.icmp sgt %i, %c-1_i32 : i32
    cf.cond_br %0, ^bb2, ^bb3
  ^bb2:
    func.call @marker() : () -> ()
    %1 = comb.add %i, %c-1_i32 : i32
    cf.br ^bb1(%1 : i32)
  ^bb3:
    llhd.yield
  }
}

// CHECK-LABEL: @RepeatLoop
hw.module @RepeatLoop() {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c3_i32 = hw.constant 3 : i32
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK-NEXT: cf.br ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: cf.br ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: llhd.yield
    cf.br ^bb1(%c3_i32 : i32)
  ^bb1(%n: i32):
    %0 = comb.icmp ne %n, %c0_i32 : i32
    cf.cond_br %0, ^bb2, ^bb3
  ^bb2:
    func.call @marker() : () -> ()
    %1 = comb.sub %n, %c1_i32 : i32
    cf.br ^bb1(%1 : i32)
  ^bb3:
    llhd.yield
  }
}

// `do ... while (i < 3)` with i from 0: the body runs for i = 0, 1, 2.
// CHECK-LABEL: @DoWhileLoop
hw.module @DoWhileLoop() {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c3_i32 = hw.constant 3 : i32
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK-NEXT: cf.br ^bb1
    // CHECK-NEXT: ^bb1:
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: func.call @marker()
    // CHECK-NEXT: cf.br ^bb2
    // CHECK-NEXT: ^bb2:
    // CHECK-NEXT: llhd.yield
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%i: i32):
    func.call @marker() : () -> ()
    %1 = comb.add %i, %c1_i32 : i32
    %0 = comb.icmp slt %1, %c3_i32 : i32
    cf.cond_br %0, ^bb1(%1 : i32), ^bb2
  ^bb2:
    llhd.yield
  }
}

// A `break` (or a `return` from an inlined function) adds an exit in the body.
// Every unrolled copy keeps its own branch to the break block, which reads the
// induction variable: the pass hands it the value as a block argument, so the
// break block takes 0, 1 or 2 and the fall-through takes 7.
// CHECK-LABEL: @LoopWithBreak
hw.module @LoopWithBreak(in %x : i3, out first : i42) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  %c7_i42 = hw.constant 7 : i42
  // CHECK: llhd.combinational
  %0 = llhd.combinational -> i42 {
    // CHECK: cf.cond_br {{%.+}}, [[BREAK:\^.+]](%c0_i42 : i42), {{\^.+}}
    // CHECK: cf.cond_br {{%.+}}, [[BREAK]](%c1_i42 : i42), {{\^.+}}
    // CHECK: cf.cond_br {{%.+}}, [[BREAK]](%c2_i42 : i42), {{\^.+}}
    // CHECK: cf.br [[END:\^.+]](%c7_i42 : i42)
    // CHECK: [[BREAK]]([[I:%.+]]: i42):
    // CHECK-NEXT: [[R:%.+]] = comb.add [[I]], %c3_i42
    // CHECK-NEXT: cf.br [[END]]([[R]] : i42)
    // CHECK: [[END]]([[OUT:%.+]]: i42):
    // CHECK-NEXT: llhd.yield [[OUT]]
    cf.br ^bb1(%c0_i42 : i42)
  ^bb1(%i: i42):
    %1 = comb.icmp slt %i, %c3_i42 : i42
    cf.cond_br %1, ^bb2, ^bb4(%c7_i42 : i42)
  ^bb2:
    %2 = comb.extract %i from 0 : (i42) -> i3
    %3 = comb.shru %x, %2 : i3
    %4 = comb.extract %3 from 0 : (i3) -> i1
    cf.cond_br %4, ^bb3, ^bb5
  ^bb3:
    %5 = comb.add %i, %c3_i42 : i42
    cf.br ^bb4(%5 : i42)
  ^bb5:
    %6 = comb.add %i, %c1_i42 : i42
    cf.br ^bb1(%6 : i42)
  ^bb4(%r: i42):
    llhd.yield %r : i42
  }
  hw.output %0 : i42
}

// The block where a break and the normal exit join reads the induction
// variable directly. All its predecessors are in the loop, so each passes the
// value as a new block argument: 0, 1 or 2 from a break, 3 at the end.
// CHECK-LABEL: @LoopWithBreakJoiningNormalExit
hw.module @LoopWithBreakJoiningNormalExit(out x : i42) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  // CHECK: llhd.combinational
  %0 = llhd.combinational -> i42 {
    // CHECK: cf.cond_br {{%.+}}, [[JOIN:\^.+]](%c0_i42 : i42), {{\^.+}}
    // CHECK: cf.cond_br {{%.+}}, [[JOIN]](%c1_i42 : i42), {{\^.+}}
    // CHECK: cf.cond_br {{%.+}}, [[JOIN]](%c2_i42 : i42), {{\^.+}}
    // CHECK: cf.br [[JOIN]](%c3_i42 : i42)
    // CHECK: [[JOIN]]([[OUT:%.+]]: i42):
    // CHECK-NEXT: llhd.yield [[OUT]]
    cf.br ^bb1(%c0_i42 : i42)
  ^bb1(%i: i42):
    %1 = comb.icmp slt %i, %c3_i42 : i42
    cf.cond_br %1, ^bb2, ^bb4
  ^bb2:
    %2 = comb.extract %i from 0 : (i42) -> i1
    cf.cond_br %2, ^bb4, ^bb5
  ^bb5:
    %6 = comb.add %i, %c1_i42 : i42
    cf.br ^bb1(%6 : i42)
  ^bb4:
    llhd.yield %i : i42
  }
  hw.output %0 : i42
}

// Here the break block and the normal exit each branch to a join block that
// reads the induction variable. The join is entered only from blocks outside
// the loop, so the value cannot be passed to it from every unrolled copy, and
// the loop is left alone (the VX backend then refuses it by name).
// CHECK-LABEL: @SkipLoopWithLoopValueAfterTwoExits
hw.module @SkipLoopWithLoopValueAfterTwoExits(out x : i42) {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  %0 = llhd.combinational -> i42 {
    // CHECK: cf.br
    // CHECK: comb.icmp slt
    // CHECK: cf.cond_br
    // CHECK: cf.cond_br
    // CHECK: comb.add
    // CHECK: cf.br
    cf.br ^bb1(%c0_i42 : i42)
  ^bb1(%i: i42):
    %1 = comb.icmp slt %i, %c3_i42 : i42
    cf.cond_br %1, ^bb2, ^bb6
  ^bb2:
    %2 = comb.extract %i from 0 : (i42) -> i1
    cf.cond_br %2, ^bb5, ^bb3
  ^bb3:
    %3 = comb.add %i, %c1_i42 : i42
    cf.br ^bb1(%3 : i42)
  ^bb5:
    cf.br ^bb4
  ^bb6:
    cf.br ^bb4
  ^bb4:
    llhd.yield %i : i42
  }
  hw.output %0 : i42
}

// More than 1024 iterations, and a loop whose counter steps past its bound
// (0, 2, 4, ... never equals 3), are left alone.
// CHECK-LABEL: @SkipLoopWithTooManyTrips
hw.module @SkipLoopWithTooManyTrips() {
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c1025_i32 = hw.constant 1025 : i32
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%i: i32):
    %0 = comb.icmp ult %i, %c1025_i32 : i32
    cf.cond_br %0, ^bb2, ^bb3
  ^bb2:
    %1 = comb.add %i, %c1_i32 : i32
    cf.br ^bb1(%1 : i32)
  ^bb3:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipNonTerminatingLoop
hw.module @SkipNonTerminatingLoop() {
  %c0_i32 = hw.constant 0 : i32
  %c2_i32 = hw.constant 2 : i32
  %c3_i32 = hw.constant 3 : i32
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^bb1(%c0_i32 : i32)
  ^bb1(%i: i32):
    %0 = comb.icmp ne %i, %c3_i32 : i32
    cf.cond_br %0, ^bb2, ^bb3
  ^bb2:
    %1 = comb.add %i, %c2_i32 : i32
    cf.br ^bb1(%1 : i32)
  ^bb3:
    llhd.yield
  }
}

// A negative bound is fine now that the trip count comes from stepping the
// induction variable: i = 3, 2, 1, 0, -1.
// CHECK-LABEL: @LoopWithNegativeBound
hw.module @LoopWithNegativeBound() {
  %c_loop_start = hw.constant  3 : i42
  %c_loop_end   = hw.constant -2 : i42
  %c_loop_inc   = hw.constant -1 : i42
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK: func.call @marker() : () -> ()
    // CHECK-NEXT: func.call @marker() : () -> ()
    // CHECK-NEXT: func.call @marker() : () -> ()
    // CHECK-NEXT: func.call @marker() : () -> ()
    // CHECK-NEXT: func.call @marker() : () -> ()
    // CHECK-NOT: func.call @marker() : () -> ()
    // CHECK: llhd.yield
    cf.br ^header(%c_loop_start: i42)
  ^header(%i: i42):
    %1 = comb.icmp sgt %i, %c_loop_end : i42
    cf.cond_br %1, ^body, ^exit
  ^body:
    func.call @marker() : () -> ()
    %iInc = comb.add %i, %c_loop_inc : i42
    cf.br ^header(%iInc: i42)
  ^exit:
    llhd.yield
  }
}
