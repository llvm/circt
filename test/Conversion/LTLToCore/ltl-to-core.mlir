// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

// CHECK: hw.module @Implication(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[C:%.+]] : !ltl.property, in [[CLK:%.+]] : i1)
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[NOT_A:%.+]] = comb.xor [[A]], [[TRUE]] : i1
// CHECK: [[OR:%.+]] = comb.or [[NOT_A]], [[B]] : i1
// CHECK: verif.assert [[OR]] : i1
// CHECK: verif.clocked_assert [[OR]], posedge [[CLK:%.+]] : i1
// CHECK: [[IMP2:%.+]] = ltl.implication [[A]], [[B]] : i1, i1
// CHECK: [[NOT_IMP2:%.+]] = ltl.not [[IMP2]] : !ltl.property
// CHECK: verif.cover [[NOT_IMP2]] : !ltl.property
// CHECK: [[IMP3:%.+]] = ltl.implication [[B]], [[C]] : i1, !ltl.property
// CHECK: verif.assert [[IMP3]] : !ltl.property

hw.module @Implication(in %a: i1, in %b: i1, in %c: !ltl.property, in %clk: i1) {
  // Convert if both operands are i1 and the only users are asserts
  %imp1 = ltl.implication %a, %b : i1, i1
  verif.assert %imp1 : !ltl.property
  verif.clocked_assert %imp1, posedge %clk : !ltl.property
  // Don't convert if there are non-assert users
  %imp2 = ltl.implication %a, %b : i1, i1
  %user = ltl.not %imp2 : !ltl.property
  verif.cover %user : !ltl.property
  // Or if there are non-i1 operands
  %imp3 = ltl.implication %b, %c : i1, !ltl.property
  verif.assert %imp3 : !ltl.property
}

// CHECK: hw.module @Not(in [[A:%.+]] : i1, in [[B:%.+]] : !ltl.property, in [[CLK:%.+]] : i1)
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[NOT_A:%.+]] = comb.xor [[A]], [[TRUE]] : i1
// CHECK: verif.assert [[NOT_A]] : i1
// CHECK: verif.clocked_assert [[NOT_A]], posedge [[CLK:%.+]] : i1
// CHECK: [[NOT2:%.+]] = ltl.not [[A]] : i1
// CHECK: [[AND:%.+]] = ltl.and
// CHECK: verif.cover [[AND]] : !ltl.property
// CHECK: [[NOT_B:%.+]] = ltl.not [[B]] : !ltl.property
// CHECK: verif.assert [[NOT_B]] : !ltl.property

hw.module @Not(in %a: i1, in %b: !ltl.property, in %clk: i1) {
  // Convert if both operands are i1 and the only users are asserts
  %not1 = ltl.not %a : i1
  verif.assert %not1 : !ltl.property
  verif.clocked_assert %not1, posedge %clk : !ltl.property
  // Don't convert if there are non-assert users
  %not2 = ltl.not %a : i1
  %user = ltl.and %not2, %not2 : !ltl.property, !ltl.property
  verif.cover %user : !ltl.property
  // Or if there are non-i1 operands
  %not3 = ltl.not %b : !ltl.property
  verif.assert %not3 : !ltl.property
}

// CHECK: hw.module @And(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[C:%.+]] : !ltl.property, in [[CLK:%.+]] : i1)
// CHECK: [[AND1:%.+]] = comb.and [[A]], [[B]] : i1
// CHECK: verif.assert [[AND1]] : i1
// CHECK: verif.clocked_assert [[AND1]], posedge [[CLK]] : i1
// CHECK: [[AND2:%.+]] = comb.and [[A]], [[B]] : i1
// CHECK: [[USER:%.+]] = hw.wire [[AND2]] : i1
// CHECK: [[AND3:%.+]] = ltl.and [[B]], [[C]] : i1, !ltl.property
// CHECK: verif.assert [[AND3]] : !ltl.property

hw.module @And(in %a: i1, in %b: i1, in %c: !ltl.property, in %clk: i1) {
  // Convert if both operands are i1 and the only users are asserts
  %and1 = ltl.and %a, %b : i1, i1
  verif.assert %and1 : i1
  verif.clocked_assert %and1, posedge %clk : i1
  // Convert if there are non-assert users but the result type is i1
  %and2 = ltl.and %a, %b : i1, i1
  %user = hw.wire %and2 : i1
  // Don't convert if there are non-i1 operands (and therefore results)
  %and3 = ltl.and %b, %c : i1, !ltl.property
  verif.assert %and3 : !ltl.property
}

// CHECK: hw.module @Or(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[C:%.+]] : !ltl.property, in [[CLK:%.+]] : i1)
// CHECK: [[OR1:%.+]] = comb.or [[A]], [[B]] : i1
// CHECK: verif.assert [[OR1]] : i1
// CHECK: verif.clocked_assert [[OR1]], posedge [[CLK]] : i1
// CHECK: [[OR2:%.+]] = comb.or [[A]], [[B]] : i1
// CHECK: [[USER:%.+]] = hw.wire [[OR2]] : i1
// CHECK: [[OR3:%.+]] = ltl.or [[B]], [[C]] : i1, !ltl.property
// CHECK: verif.assert [[OR3]] : !ltl.property

hw.module @Or(in %a: i1, in %b: i1, in %c: !ltl.property, in %clk: i1) {
  // Convert if both operands are i1 and the only users are asserts
  %or1 = ltl.or %a, %b : i1, i1
  verif.assert %or1 : i1
  verif.clocked_assert %or1, posedge %clk : i1
  // Convert if there are non-assert users but the result type is i1
  %or2 = ltl.or %a, %b : i1, i1
  %user = hw.wire %or2 : i1
  // Don't convert if there are non-i1 operands (and therefore results)
  %or3 = ltl.or %b, %c : i1, !ltl.property
  verif.assert %or3 : !ltl.property
}

// CHECK: hw.module @Intersect(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[C:%.+]] : !ltl.sequence, in [[CLK:%.+]] : i1)
// CHECK: [[INT1:%.+]] = comb.and [[A]], [[B]] : i1
// CHECK: verif.assert [[INT1]] : i1
// CHECK: verif.clocked_assert [[INT1]], posedge [[CLK]] : i1
// CHECK: [[INT2:%.+]] = comb.and [[A]], [[B]] : i1
// CHECK: [[USER:%.+]] = hw.wire [[INT2]] : i1
// CHECK: [[INT3:%.+]] = ltl.intersect [[B]], [[C]] : i1, !ltl.sequence
// CHECK: verif.assert [[INT3]] : !ltl.sequence

hw.module @Intersect(in %a: i1, in %b: i1, in %c: !ltl.sequence, in %clk: i1) {
  // Boolean intersection is instantaneous, i.e. conjunction. Convert if both
  // operands are i1 and the only users are asserts
  %int1 = ltl.intersect %a, %b : i1, i1
  verif.assert %int1 : i1
  verif.clocked_assert %int1, posedge %clk : i1
  // Convert if there are non-assert users but the result type is i1
  %int2 = ltl.intersect %a, %b : i1, i1
  %user = hw.wire %int2 : i1
  // Don't convert if there are non-i1 operands (and therefore results)
  %int3 = ltl.intersect %b, %c : i1, !ltl.sequence
  verif.assert %int3 : !ltl.sequence
}

// CHECK: hw.module @Past(in [[A:%.+]] : i32, in [[CLK:%.+]] : i1)
// CHECK: [[TOCLK1:%.+]] = seq.to_clock [[CLK]]
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[REG1:%.+]] = seq.shiftreg[1] [[A]], [[TOCLK1]], [[TRUE]] : i32
// CHECK: [[TOCLK2:%.+]] = seq.to_clock [[CLK]]
// CHECK: [[TRUE1:%.+]] = hw.constant true
// CHECK: [[REG2:%.+]] = seq.shiftreg[5] [[A]], [[TOCLK2]], [[TRUE1]] : i32

hw.module @Past(in %a: i32, in %clk: i1) {
  ltl.past %a, 1 clk %clk : i32
  ltl.past %a, 5 clk %clk : i32
}

// CHECK-LABEL: hw.module @ClockedAtom(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[CLOCK]] initial
// CHECK:         %[[FALSE:.+]] = hw.constant false
// CHECK:         %[[DONT_CARE_INIT:.+]] = seq.initial()
// CHECK:           %[[TRUE:.+]] = hw.constant true
// CHECK:           seq.yield %[[TRUE]] : i1
// CHECK:         %[[DONT_CARE:.+]] = seq.compreg %[[FALSE]], %[[CLOCK]] initial %[[DONT_CARE_INIT]]
// CHECK:         %[[GUARDED:.+]] = comb.or %[[DONT_CARE]], %[[SAMPLED_A]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "clocked_atom" : i1
hw.module @ClockedAtom(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  verif.assert %1 label "clocked_atom" : !ltl.sequence
  hw.output
}

// CHECK-LABEL: hw.module @NegedgeClock(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK:         %[[CLOCK_I1:.+]] = seq.from_clock %[[CLOCK]]
// CHECK:         %[[CLOCK_TRUE:.+]] = hw.constant true
// CHECK:         %[[NOT_CLOCK:.+]] = comb.xor %[[CLOCK_I1]], %[[CLOCK_TRUE]] : i1
// CHECK:         %[[NEGEDGE_CLOCK:.+]] = seq.to_clock %[[NOT_CLOCK]]
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[NEGEDGE_CLOCK]] initial
// CHECK:         %[[FALSE:.+]] = hw.constant false
// CHECK:         %[[DONT_CARE_TRUE:.+]] = hw.constant true
// CHECK:         %[[DONT_CARE_NOT_CLOCK:.+]] = comb.xor %[[CLOCK_I1]], %[[DONT_CARE_TRUE]] : i1
// CHECK:         %[[DONT_CARE_CLOCK:.+]] = seq.to_clock %[[DONT_CARE_NOT_CLOCK]]
// CHECK:         %[[DONT_CARE_INIT:.+]] = seq.initial()
// CHECK:           %[[TRUE:.+]] = hw.constant true
// CHECK:           seq.yield %[[TRUE]] : i1
// CHECK:         %[[DONT_CARE:.+]] = seq.compreg %[[FALSE]], %[[DONT_CARE_CLOCK]] initial %[[DONT_CARE_INIT]]
// CHECK:         %[[GUARDED:.+]] = comb.or %[[DONT_CARE]], %[[SAMPLED_A]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "negedge" : i1
hw.module @NegedgeClock(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, negedge %0 : i1
  verif.assert %1 label "negedge" : !ltl.sequence
  hw.output
}

// CHECK-LABEL: hw.module @BothEdgeClock(
// CHECK-NOT:     seq.compreg
// CHECK:         %[[ATOM:.+]] = ltl.clocked_atom
// CHECK:         verif.assert %[[ATOM]] : !ltl.sequence
hw.module @BothEdgeClock(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, edge %0 : i1
  verif.assert %1 : !ltl.sequence
  hw.output
}

// Boolean logic is lowered when used by assumes.
// CHECK-LABEL: hw.module @Assume
// CHECK:         [[TRUE0:%.+]] = hw.constant true
// CHECK:         [[NOT_A0:%.+]] = comb.xor %a, [[TRUE0]] : i1
// CHECK:         [[IMPL0:%.+]] = comb.or [[NOT_A0]], %b : i1
// CHECK:         verif.clocked_assume [[IMPL0]] if %en, posedge %clk label "clocked" : i1
// CHECK:         [[TRUE1:%.+]] = hw.constant true
// CHECK:         [[NOT_A1:%.+]] = comb.xor %a, [[TRUE1]] : i1
// CHECK:         [[IMPL1:%.+]] = comb.or [[NOT_A1]], %c : i1
// CHECK:         verif.assume [[IMPL1]] if %en label "unclocked" : i1
// CHECK-NOT:     ltl.
hw.module @Assume(in %a: i1, in %b: i1, in %c: i1, in %en: i1, in %clk: i1) {
  %0 = ltl.implication %a, %b : i1, i1
  verif.clocked_assume %0 if %en, posedge %clk label "clocked" : !ltl.property
  %1 = ltl.implication %a, %c : i1, i1
  verif.assume %1 if %en label "unclocked" : !ltl.property
  hw.output
}

// Boolean implications are lowered when nested in other implications:
// a |-> (b |-> c) == !a || (!b || c).
// CHECK-LABEL: hw.module @NestedImplications
// CHECK:         [[NOT_B:%.+]] = comb.xor %b, %{{.+}} : i1
// CHECK:         [[INNER:%.+]] = comb.or [[NOT_B]], %c : i1
// CHECK:         [[NOT_A:%.+]] = comb.xor %a, %{{.+}} : i1
// CHECK:         [[OUTER:%.+]] = comb.or [[NOT_A]], [[INNER]] : i1
// CHECK:         verif.clocked_assert [[OUTER]], posedge %clk : i1
// CHECK-NOT:     ltl.
hw.module @NestedImplications(in %a: i1, in %b: i1, in %c: i1, in %clk: i1) {
  %0 = ltl.implication %b, %c : i1, i1
  %1 = ltl.implication %a, %0 : i1, !ltl.property
  verif.clocked_assert %1, posedge %clk : !ltl.property
  hw.output
}

// Nested implications with a shared subexpression feeding both an assert and an
// assume.
// not((a |-> b) and (b |-> c)) or (a |-> b).
// CHECK-LABEL: hw.module @MixedAssertAndAssume
// CHECK-DAG:     [[NOT_A:%.+]] = comb.xor %a, %{{.+}} : i1
// CHECK-DAG:     [[AB:%.+]] = comb.or [[NOT_A]], %b : i1
// CHECK-DAG:     [[NOT_B:%.+]] = comb.xor %b, %{{.+}} : i1
// CHECK-DAG:     [[BC:%.+]] = comb.or [[NOT_B]], %c : i1
// CHECK-DAG:     [[AND:%.+]] = comb.and [[AB]], [[BC]] : i1
// CHECK-DAG:     [[NOT:%.+]] = comb.xor [[AND]], %{{.+}} : i1
// CHECK-DAG:     [[OR:%.+]] = comb.or [[NOT]], [[AB]] : i1
// CHECK-DAG:     verif.assert [[OR]] : i1
// CHECK-DAG:     verif.assume [[AB]] : i1
// CHECK-NOT:     ltl.
hw.module @MixedAssertAndAssume(in %a: i1, in %b: i1, in %c: i1) {
  %0 = ltl.implication %a, %b : i1, i1
  %1 = ltl.implication %b, %c : i1, i1
  %2 = ltl.and %0, %1 : !ltl.property, !ltl.property
  %3 = ltl.not %2 : !ltl.property
  %4 = ltl.or %3, %0 : !ltl.property, !ltl.property
  verif.assert %4 : !ltl.property
  verif.assume %0 : !ltl.property
  hw.output
}

// A nested chain with a non-assert/assume user is left untouched as a whole,
// including the inner implication.
// CHECK-LABEL: hw.module @NestedImplicationsNonAssertUser
// CHECK-NOT:     comb.
// CHECK:         [[INNER:%.+]] = ltl.implication %b, %c : i1, i1
// CHECK:         [[OUTER:%.+]] = ltl.implication %a, [[INNER]] : i1, !ltl.property
// CHECK:         [[CLOCKED:%.+]] = ltl.clock [[OUTER]], posedge %clk : !ltl.property
// CHECK:         verif.assert [[CLOCKED]] : !ltl.property
hw.module @NestedImplicationsNonAssertUser(in %a: i1, in %b: i1, in %c: i1, in %clk: i1) {
  %0 = ltl.implication %b, %c : i1, i1
  %1 = ltl.implication %a, %0 : i1, !ltl.property
  %2 = ltl.clock %1, posedge %clk : !ltl.property
  verif.assert %2 : !ltl.property
  hw.output
}

// A nested chain with a temporal leaf is left untouched as a whole, including
// the purely boolean implication it shares with an assert.
// CHECK-LABEL: hw.module @NestedImplicationsTemporalLeaf
// CHECK-NOT:     comb.
// CHECK:         [[INNER:%.+]] = ltl.implication %b, %c : i1, i1
// CHECK:         [[DELAY:%.+]] = ltl.delay %a, 1 : i1
// CHECK:         [[OUTER:%.+]] = ltl.implication [[DELAY]], [[INNER]] : !ltl.sequence, !ltl.property
// CHECK:         verif.assert [[OUTER]] : !ltl.property
// CHECK:         verif.assert [[INNER]] : !ltl.property
hw.module @NestedImplicationsTemporalLeaf(in %a: i1, in %b: i1, in %c: i1) {
  %0 = ltl.implication %b, %c : i1, i1
  %1 = ltl.delay %a, 1 : i1
  %2 = ltl.implication %1, %0 : !ltl.sequence, !ltl.property
  verif.assert %2 : !ltl.property
  verif.assert %0 : !ltl.property
  hw.output
}
