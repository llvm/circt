// RUN: circt-opt %s --canonicalize | FileCheck %s

func.func private @Bool(%arg0: i1)
func.func private @Seq(%arg0: !ltl.sequence)
func.func private @Prop(%arg0: !ltl.property)

// CHECK-LABEL: @DelayFolds
func.func @DelayFolds(%arg0: !ltl.sequence, %arg1: i1) {
  // delay(s, 0, 0) -> s
  // delay(i, 0, 0) -> delay(i, 0, 0)
  // CHECK-NEXT: {{%.*}} = hw.constant true
  // CHECK-NEXT: ltl.delay {{%.*}}, posedge, %arg1, 0, 0 : i1
  // CHECK-NEXT: call @Seq(%arg0)
  // CHECK-NEXT: call @Seq({{%.+}})
  %clk = hw.constant true
  %0 = ltl.delay %clk, posedge, %arg0, 0, 0 : !ltl.sequence
  %n0 = ltl.delay %clk, posedge, %arg1, 0, 0 : i1
  call @Seq(%0) : (!ltl.sequence) -> ()
  call @Seq(%n0) : (!ltl.sequence) -> ()

  // delay(delay(s, 1), 2) -> delay(s, 3)
  // CHECK-NEXT: ltl.delay {{%.*}}, posedge, %arg0, 3 :
  // CHECK-NEXT: call
  %1 = ltl.delay %clk, posedge, %arg0, 1 : !ltl.sequence
  %2 = ltl.delay %clk, posedge, %1, 2 : !ltl.sequence
  call @Seq(%2) : (!ltl.sequence) -> ()

  // delay(delay(s, 1, N), 2) -> delay(s, 3)
  // N is dropped
  // CHECK-NEXT: ltl.delay {{%.*}}, posedge, {{%.+}}, 3 :
  // CHECK-NEXT: ltl.delay {{%.*}}, posedge, {{%.+}}, 3 :
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  %3 = ltl.delay %clk, posedge, %arg0, 1, 0 : !ltl.sequence
  %4 = ltl.delay %clk, posedge, %arg0, 1, 42 : !ltl.sequence
  %5 = ltl.delay %clk, posedge, %3, 2 : !ltl.sequence
  %6 = ltl.delay %clk, posedge, %4, 2 : !ltl.sequence
  call @Seq(%5) : (!ltl.sequence) -> ()
  call @Seq(%6) : (!ltl.sequence) -> ()

  // delay(delay(s, 1), 2, N) -> delay(s, 3)
  // N is dropped
  // CHECK-NEXT: ltl.delay {{%.*}}, posedge, %arg0, 3 :
  // CHECK-NEXT: ltl.delay {{%.*}}, posedge, %arg0, 3 :
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  %7 = ltl.delay %clk, posedge, %arg0, 1 : !ltl.sequence
  %8 = ltl.delay %clk, posedge, %arg0, 1 : !ltl.sequence
  %9 = ltl.delay %clk, posedge, %7, 2, 0 : !ltl.sequence
  %10 = ltl.delay %clk, posedge, %8, 2, 42 : !ltl.sequence
  call @Seq(%9) : (!ltl.sequence) -> ()
  call @Seq(%10) : (!ltl.sequence) -> ()

  // delay(delay(s, 1, 2), 3, 0) -> delay(s, 4, 2)
  // delay(delay(s, 1, 2), 3, 5) -> delay(s, 4, 7)
  // CHECK-NEXT: ltl.delay {{%.*}}, posedge, %arg0, 4, 2 :
  // CHECK-NEXT: ltl.delay {{%.*}}, posedge, %arg0, 4, 7 :
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  %11 = ltl.delay %clk, posedge, %arg0, 1, 2 : !ltl.sequence
  %12 = ltl.delay %clk, posedge, %arg0, 1, 2 : !ltl.sequence
  %13 = ltl.delay %clk, posedge, %11, 3, 0 : !ltl.sequence
  %14 = ltl.delay %clk, posedge, %12, 3, 5 : !ltl.sequence
  call @Seq(%13) : (!ltl.sequence) -> ()
  call @Seq(%14) : (!ltl.sequence) -> ()
  return
}

// CHECK-LABEL: @ConcatFolds
func.func @ConcatFolds(%arg0: !ltl.sequence, %arg1: !ltl.sequence, %arg2: !ltl.sequence) {
  // concat(s) -> s
  // CHECK-NEXT: {{%.*}} = hw.constant true
  // CHECK-NEXT: call @Seq(%arg0)
  %clk = hw.constant true
  %0 = ltl.concat %arg0 : !ltl.sequence
  call @Seq(%0) : (!ltl.sequence) -> ()

  // concat(concat(s0, s1), s2) -> concat(s0, s1, s2)
  // concat(s0, concat(s1, s2)) -> concat(s0, s1, s2)
  // concat(concat(s0, s1), s2, s0, concat(s1, s2)) -> concat(s0, s1, s2, s0, s1, s2)
  // CHECK-NEXT: ltl.concat %arg0, %arg1, %arg2 :
  // CHECK-NEXT: ltl.concat %arg0, %arg1, %arg2 :
  // CHECK-NEXT: ltl.concat %arg0, %arg1, %arg2, %arg0, %arg1, %arg2 :
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  // CHECK-NEXT: call
  %1 = ltl.concat %arg0, %arg1 : !ltl.sequence, !ltl.sequence
  %2 = ltl.concat %1, %arg2 : !ltl.sequence, !ltl.sequence
  %3 = ltl.concat %arg1, %arg2 : !ltl.sequence, !ltl.sequence
  %4 = ltl.concat %arg0, %3 : !ltl.sequence, !ltl.sequence
  %5 = ltl.concat %1, %arg2, %arg0, %3 : !ltl.sequence, !ltl.sequence, !ltl.sequence, !ltl.sequence
  call @Seq(%2) : (!ltl.sequence) -> ()
  call @Seq(%4) : (!ltl.sequence) -> ()
  call @Seq(%5) : (!ltl.sequence) -> ()

  // delay(concat(s0, s1), N, M) -> concat(delay(s0, N, M), s1)
  // CHECK-NEXT: [[TMP:%.+]] = ltl.delay {{%.*}}, posedge, %arg0, 2, 3 :
  // CHECK-NEXT: ltl.concat [[TMP]], %arg1 :
  // CHECK-NEXT: call
  %6 = ltl.concat %arg0, %arg1 : !ltl.sequence, !ltl.sequence
  %7 = ltl.delay %clk, posedge, %6, 2, 3 : !ltl.sequence
  call @Seq(%7) : (!ltl.sequence) -> ()
  return
}

// CHECK-LABEL: @RepeatFolds
func.func @RepeatFolds(%arg0: !ltl.sequence) {
  // repeat(s, 1, 0) -> s
  // CHECK-NEXT: call @Seq(%arg0)
  %0 = ltl.repeat %arg0, 1, 0: !ltl.sequence
  call @Seq(%0) : (!ltl.sequence) -> ()

  return
}

// CHECK-LABEL: @GoToRepeatFolds
func.func @GoToRepeatFolds(%arg0: !ltl.sequence) {
  // repeat(s, 1, 0) -> s
  // CHECK-NEXT: call @Seq(%arg0)
  %0 = ltl.goto_repeat %arg0, 1, 0: !ltl.sequence
  call @Seq(%0) : (!ltl.sequence) -> ()

  return
}

// CHECK-LABEL: @NonConsecutiveRepeatFolds
func.func @NonConsecutiveRepeatFolds(%arg0: !ltl.sequence) {
  // repeat(s, 1, 0) -> s
  // CHECK-NEXT: call @Seq(%arg0)
  %0 = ltl.non_consecutive_repeat %arg0, 1, 0: !ltl.sequence
  call @Seq(%0) : (!ltl.sequence) -> ()

  return
}

// CHECK-LABEL: @CanonicalizeToComb
func.func @CanonicalizeToComb(%arg0: i1, %arg1: i1, %arg2: i1) {
  // CHECK-NEXT: comb.and bin %arg0, %arg1, %arg2 : i1
  %0 = ltl.and %arg0, %arg1, %arg2 : i1, i1, i1
  // CHECK-NEXT: comb.or bin %arg0, %arg1, %arg2 : i1
  %1 = ltl.or %arg0, %arg1, %arg2 : i1, i1, i1
  // CHECK-NEXT: comb.and bin %arg0, %arg1, %arg2 : i1
  %2 = ltl.intersect %arg0, %arg1, %arg2 : i1, i1, i1

  call @Bool(%0) : (i1) -> ()
  call @Bool(%1) : (i1) -> ()
  call @Bool(%2) : (i1) -> ()
  return
}

// CHECK-LABEL: @ImplicationFolds
// CHECK-SAME: (%[[A:.+]]: i1)
func.func @ImplicationFolds(%a: i1) -> (!ltl.property, !ltl.property) {
  %false = hw.constant false
  %true = hw.constant true

  // implication(false, x) -> boolean_constant(true)
  // implication(x, true) -> boolean_constant(true)
  // Both fold to the same constant, which gets CSE'd
  // CHECK: %[[PROP:.+]] = ltl.boolean_constant true
  // CHECK-NOT: ltl.implication
  %0 = ltl.implication %false, %a : i1, i1
  %1 = ltl.implication %a, %true : i1, i1

  // CHECK: return %[[PROP]], %[[PROP]]
  return %0, %1 : !ltl.property, !ltl.property
}
