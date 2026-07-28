// RUN: circt-opt %s --canonicalize | FileCheck %s

func.func private @Bool(%arg0: i1)
func.func private @Seq(%arg0: !ltl.sequence)
func.func private @Prop(%arg0: !ltl.property)

// CHECK-LABEL: @ClockedDelayFolds
// CHECK-SAME: (%[[S:.+]]: !ltl.sequence, %[[I:.+]]: i1, %[[CLK:.+]]: i1)
func.func @ClockedDelayFolds(%arg0: !ltl.sequence, %arg1: i1, %clk: i1) {
  // Clocked delays must not fold away, even for a zero delay, because doing so
  // would drop their explicit clock.
  // CHECK-NEXT: %[[SAME:.+]] = ltl.clocked_delay %[[S]], posedge %[[CLK]], 0, 0 : !ltl.sequence
  // CHECK-NEXT: %[[D:.+]] = ltl.clocked_delay %[[I]], posedge %[[CLK]], 0, 0 : i1
  // CHECK-NEXT: call @Seq(%[[SAME]])
  // CHECK-NEXT: call @Seq(%[[D]])
  %0 = ltl.clocked_delay %arg0, posedge %clk, 0, 0 : !ltl.sequence
  %n0 = ltl.clocked_delay %arg1, posedge %clk, 0, 0 : i1
  call @Seq(%0) : (!ltl.sequence) -> ()
  call @Seq(%n0) : (!ltl.sequence) -> ()

  // Nested clocked delays with same clock/edge: merge delays
  // clocked_delay(clocked_delay(s, posedge clk, 1), posedge clk, 2)
  //   -> clocked_delay(s, posedge clk, 3)
  // CHECK-NEXT: %[[D:.+]] = ltl.clocked_delay %[[S]], posedge %[[CLK]], 3 :
  // CHECK-NEXT: call @Seq(%[[D]])
  %1 = ltl.clocked_delay %arg0, posedge %clk, 1 : !ltl.sequence
  %2 = ltl.clocked_delay %1, posedge %clk, 2 : !ltl.sequence
  call @Seq(%2) : (!ltl.sequence) -> ()

  // Inner has length, outer does not: length dropped
  // clocked_delay(clocked_delay(s, posedge clk, 1, 42), posedge clk, 2)
  //   -> clocked_delay(s, posedge clk, 3)
  // CHECK-NEXT: %[[D:.+]] = ltl.clocked_delay %[[S]], posedge %[[CLK]], 3 :
  // CHECK-NEXT: call @Seq(%[[D]])
  %3 = ltl.clocked_delay %arg0, posedge %clk, 1, 42 : !ltl.sequence
  %4 = ltl.clocked_delay %3, posedge %clk, 2 : !ltl.sequence
  call @Seq(%4) : (!ltl.sequence) -> ()

  // Outer has length, inner does not: length dropped
  // clocked_delay(clocked_delay(s, posedge clk, 1), posedge clk, 2, 5)
  //   -> clocked_delay(s, posedge clk, 3)
  // CHECK-NEXT: %[[D:.+]] = ltl.clocked_delay %[[S]], posedge %[[CLK]], 3 :
  // CHECK-NEXT: call @Seq(%[[D]])
  %5 = ltl.clocked_delay %arg0, posedge %clk, 1 : !ltl.sequence
  %6 = ltl.clocked_delay %5, posedge %clk, 2, 5 : !ltl.sequence
  call @Seq(%6) : (!ltl.sequence) -> ()

  // Both have length: lengths merged
  // clocked_delay(clocked_delay(s, posedge clk, 1, 2), posedge clk, 3, 5)
  //   -> clocked_delay(s, posedge clk, 4, 7)
  // CHECK-NEXT: %[[D:.+]] = ltl.clocked_delay %[[S]], posedge %[[CLK]], 4, 7 :
  // CHECK-NEXT: call @Seq(%[[D]])
  %7 = ltl.clocked_delay %arg0, posedge %clk, 1, 2 : !ltl.sequence
  %8 = ltl.clocked_delay %7, posedge %clk, 3, 5 : !ltl.sequence
  call @Seq(%8) : (!ltl.sequence) -> ()

  // Both have length, outer length is 0: no drop
  // clocked_delay(clocked_delay(s, posedge clk, 1, 2), posedge clk, 3, 0)
  //   -> clocked_delay(s, posedge clk, 4, 2)
  // CHECK-NEXT: %[[D:.+]] = ltl.clocked_delay %[[S]], posedge %[[CLK]], 4, 2 :
  // CHECK-NEXT: call @Seq(%[[D]])
  %9 = ltl.clocked_delay %arg0, posedge %clk, 1, 2 : !ltl.sequence
  %10 = ltl.clocked_delay %9, posedge %clk, 3, 0 : !ltl.sequence
  call @Seq(%10) : (!ltl.sequence) -> ()

  // Different edge: should NOT merge
  // CHECK-NEXT: %[[D1:.+]] = ltl.clocked_delay %[[S]], posedge %[[CLK]], 1 :
  // CHECK-NEXT: %[[D2:.+]] = ltl.clocked_delay %[[D1]], negedge %[[CLK]], 2 :
  // CHECK-NEXT: call @Seq(%[[D2]])
  %11 = ltl.clocked_delay %arg0, posedge %clk, 1 : !ltl.sequence
  %12 = ltl.clocked_delay %11, negedge %clk, 2 : !ltl.sequence
  call @Seq(%12) : (!ltl.sequence) -> ()

  // Different clock: should NOT merge
  // CHECK-NEXT: %[[D1:.+]] = ltl.clocked_delay %[[S]], posedge %[[CLK]], 1 :
  // CHECK-NEXT: %[[D2:.+]] = ltl.clocked_delay %[[D1]], posedge %[[I]], 2 :
  // CHECK-NEXT: call @Seq(%[[D2]])
  %13 = ltl.clocked_delay %arg0, posedge %clk, 1 : !ltl.sequence
  %14 = ltl.clocked_delay %13, posedge %arg1, 2 : !ltl.sequence
  call @Seq(%14) : (!ltl.sequence) -> ()

  return
}

// CHECK-LABEL: @ConcatFolds
// CHECK-SAME: (%[[S0:.+]]: !ltl.sequence, %[[S1:.+]]: !ltl.sequence, %[[S2:.+]]: !ltl.sequence, %[[CLK:.+]]: i1)
func.func @ConcatFolds(%arg0: !ltl.sequence, %arg1: !ltl.sequence, %arg2: !ltl.sequence, %clk: i1) {
  // concat(s) -> s
  // CHECK-NEXT: call @Seq(%[[S0]])
  %0 = ltl.concat %arg0 : !ltl.sequence
  call @Seq(%0) : (!ltl.sequence) -> ()

  // concat(concat(s0, s1), s2) -> concat(s0, s1, s2)
  // concat(s0, concat(s1, s2)) -> concat(s0, s1, s2)
  // concat(concat(s0, s1), s2, s0, concat(s1, s2)) -> concat(s0, s1, s2, s0, s1, s2)
  // CHECK-NEXT: ltl.concat %[[S0]], %[[S1]], %[[S2]] :
  // CHECK-NEXT: ltl.concat %[[S0]], %[[S1]], %[[S2]] :
  // CHECK-NEXT: ltl.concat %[[S0]], %[[S1]], %[[S2]], %[[S0]], %[[S1]], %[[S2]] :
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

  // clocked_delay(concat(s0, s1), posedge clk, N, M)
  //   -> concat(clocked_delay(s0, posedge clk, N, M), s1)
  // CHECK-NEXT: %[[DELAYED:.+]] = ltl.clocked_delay %[[S0]], posedge %[[CLK]], 2, 3 :
  // CHECK-NEXT: %[[CONCAT:.+]] = ltl.concat %[[DELAYED]], %[[S1]] :
  // CHECK-NEXT: call @Seq(%[[CONCAT]])
  %8 = ltl.concat %arg0, %arg1 : !ltl.sequence, !ltl.sequence
  %9 = ltl.clocked_delay %8, posedge %clk, 2, 3 : !ltl.sequence
  call @Seq(%9) : (!ltl.sequence) -> ()
  return
}

// CHECK-LABEL: @ClockedRepeatFold
// CHECK: %[[REPEAT:.+]] = ltl.clocked_repeat %arg0, posedge %{{.+}}, 1, 0
// CHECK: return %[[REPEAT]]
func.func @ClockedRepeatFold(%s: !ltl.sequence, %clk: i1) -> !ltl.sequence {
  %0 = ltl.clocked_repeat %s, posedge %clk, 1, 0 : !ltl.sequence
  return %0 : !ltl.sequence
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
