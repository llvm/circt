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

// CHECK-LABEL: @SkipLoopWithMultipleExits
hw.module @SkipLoopWithMultipleExits() {
  %c0_i42 = hw.constant 0 : i42
  %c1_i42 = hw.constant 1 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
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
    %2 = comb.sub %0, %c1_i42 : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
}

// CHECK-LABEL: @SkipLoopWithUnsupportedBounds
hw.module @SkipLoopWithUnsupportedBounds() {
  %c1_i42 = hw.constant 1 : i42
  %c2_i42 = hw.constant 2 : i42
  %c3_i42 = hw.constant 3 : i42
  llhd.combinational {
    // CHECK: cf.br
    // CHECK: cf.cond_br
    // CHECK: cf.br
    // CHECK: llhd.yield
    cf.br ^header(%c1_i42 : i42)
  ^header(%0: i42):
    %1 = comb.icmp slt %0, %c3_i42 : i42
    cf.cond_br %1, ^body, ^exit
  ^body:
    %2 = comb.add %0, %c2_i42 : i42
    cf.br ^header(%2 : i42)
  ^exit:
    llhd.yield
  }
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
