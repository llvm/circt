// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s --implicit-check-not=ltl

// CHECK: hw.module @Implication(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[CLK:%.+]] : i1)
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[NOT_A:%.+]] = comb.xor [[A]], [[TRUE]] : i1
// CHECK: [[OR:%.+]] = comb.or [[NOT_A]], [[B]] : i1
// CHECK: verif.assert [[OR]] : i1
// CHECK: verif.clocked_assert [[OR]], posedge [[CLK:%.+]] : i1

hw.module @Implication(in %a: i1, in %b: i1, in %clk: i1) {
  %imp1 = ltl.implication %a, %b : i1, i1
  verif.assert %imp1 : !ltl.property
  verif.clocked_assert %imp1, posedge %clk : !ltl.property
}

// CHECK: hw.module @Not(in [[A:%.+]] : i1, in [[CLK:%.+]] : i1)
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[NOT_A:%.+]] = comb.xor [[A]], [[TRUE]] : i1
// CHECK: verif.assert [[NOT_A]] : i1
// CHECK: verif.clocked_assert [[NOT_A]], posedge [[CLK:%.+]] : i1

hw.module @Not(in %a: i1, in %clk: i1) {
  %not1 = ltl.not %a : i1
  verif.assert %not1 : !ltl.property
  verif.clocked_assert %not1, posedge %clk : !ltl.property
}

// CHECK: hw.module @And(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[CLK:%.+]] : i1)
// CHECK: [[AND1:%.+]] = comb.and [[A]], [[B]] : i1
// CHECK: verif.assert [[AND1]] : i1
// CHECK: verif.clocked_assert [[AND1]], posedge [[CLK]] : i1
// CHECK: [[AND2:%.+]] = comb.and [[A]], [[B]] : i1
// CHECK: [[USER:%.+]] = hw.wire [[AND2]] : i1

hw.module @And(in %a: i1, in %b: i1, in %clk: i1) {
  %and1 = ltl.and %a, %b : i1, i1
  verif.assert %and1 : i1
  verif.clocked_assert %and1, posedge %clk : i1
  %and2 = ltl.and %a, %b : i1, i1
  %user = hw.wire %and2 : i1
}

// CHECK: hw.module @Or(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[CLK:%.+]] : i1)
// CHECK: [[OR1:%.+]] = comb.or [[A]], [[B]] : i1
// CHECK: verif.assert [[OR1]] : i1
// CHECK: verif.clocked_assert [[OR1]], posedge [[CLK]] : i1
// CHECK: [[OR2:%.+]] = comb.or [[A]], [[B]] : i1
// CHECK: [[USER:%.+]] = hw.wire [[OR2]] : i1

hw.module @Or(in %a: i1, in %b: i1, in %clk: i1) {
  %or1 = ltl.or %a, %b : i1, i1
  verif.assert %or1 : i1
  verif.clocked_assert %or1, posedge %clk : i1
  %or2 = ltl.or %a, %b : i1, i1
  %user = hw.wire %or2 : i1
}

// CHECK: hw.module @Intersect(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[CLK:%.+]] : i1)
// CHECK: [[INT1:%.+]] = comb.and [[A]], [[B]] : i1
// CHECK: verif.assert [[INT1]] : i1
// CHECK: verif.clocked_assert [[INT1]], posedge [[CLK]] : i1
// CHECK: [[INT2:%.+]] = comb.and [[A]], [[B]] : i1
// CHECK: [[USER:%.+]] = hw.wire [[INT2]] : i1

hw.module @Intersect(in %a: i1, in %b: i1, in %clk: i1) {
  %int1 = ltl.intersect %a, %b : i1, i1
  verif.assert %int1 : i1
  verif.clocked_assert %int1, posedge %clk : i1
  %int2 = ltl.intersect %a, %b : i1, i1
  %user = hw.wire %int2 : i1
}

// CHECK-LABEL: hw.module @BooleanConstant
// CHECK:         %[[TRUE:.+]] = hw.constant true
// CHECK:         verif.assert %[[TRUE]] : i1
// CHECK:         %[[FALSE:.+]] = hw.constant false
// CHECK:         verif.assume %[[FALSE]] : i1
hw.module @BooleanConstant() {
  %true = ltl.boolean_constant true
  verif.assert %true : !ltl.property
  %false = ltl.boolean_constant false
  verif.assume %false : !ltl.property
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

<<<<<<< Updated upstream
// CHECK-LABEL: hw.module @ClockedAtom(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[CLOCK]] initial
// CHECK:         %[[PROPERTY:.+]] = comb.or %[[SAMPLED_A]] : i1
// CHECK:         %[[VALID:.+]] = seq.compreg %{{.+}}, %[[CLOCK]] initial
// CHECK:         %[[VALID_PROPERTY:.+]] = comb.and %[[VALID]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID_PROPERTY]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[PROPERTY]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "clocked_atom" : i1
hw.module @ClockedAtom(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  verif.assert %1 label "clocked_atom" : !ltl.sequence
  hw.output
}

// CHECK-LABEL: hw.module @ImplicationDelay(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK-SAME:    in %[[B:[^, ]+]] : i1
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[CLOCK]] initial
// CHECK:         %[[SAMPLED_B:.+]] = seq.compreg %[[B]], %[[CLOCK]] initial
// CHECK:         %[[CONSEQUENT:.+]] = comb.or %[[SAMPLED_B]] : i1
// CHECK:         %[[PAST_A_1:.+]] = seq.compreg %[[SAMPLED_A]], %[[CLOCK]] initial
// CHECK:         %[[PAST_A_2:.+]] = seq.compreg %[[PAST_A_1]], %[[CLOCK]] initial
// CHECK:         %[[NOT_A:.+]] = comb.xor %[[PAST_A_2]], %{{.+}} : i1
// CHECK:         %[[IMPLICATION:.+]] = comb.or %[[NOT_A]], %[[CONSEQUENT]] : i1
// CHECK:         %[[PROPERTY:.+]] = comb.and %[[IMPLICATION]] : i1
// CHECK:         %[[VALID_0:.+]] = seq.compreg %{{.+}}, %[[CLOCK]] initial
// CHECK:         %[[VALID_1:.+]] = seq.compreg %[[VALID_0]], %[[CLOCK]] initial
// CHECK:         %[[VALID_2:.+]] = seq.compreg %[[VALID_1]], %[[CLOCK]] initial
// CHECK:         %[[VALID_PROPERTY:.+]] = comb.and %[[VALID_2]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID_PROPERTY]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[PROPERTY]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "implication_delay" : i1
hw.module @ImplicationDelay(in %clock : !seq.clock, in %a : i1, in %b : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_atom %b, posedge %0 : i1
  %3 = ltl.clocked_delay %2, posedge %0, 2, 0 : !ltl.sequence
  %4 = ltl.implication %1, %3 : !ltl.sequence, !ltl.sequence
  verif.assert %4 label "implication_delay" : !ltl.property
  hw.output
}

// CHECK-LABEL: hw.module @DirectDelay(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[CLOCK]] initial
// CHECK:         %[[PROPERTY:.+]] = comb.or %[[SAMPLED_A]] : i1
// CHECK:         %[[VALID_0:.+]] = seq.compreg %{{.+}}, %[[CLOCK]] initial
// CHECK:         %[[VALID_1:.+]] = seq.compreg %[[VALID_0]], %[[CLOCK]] initial
// CHECK:         %[[VALID_2:.+]] = seq.compreg %[[VALID_1]], %[[CLOCK]] initial
// CHECK:         %[[VALID_PROPERTY:.+]] = comb.and %[[VALID_2]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID_PROPERTY]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[PROPERTY]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "direct_delay" : i1
hw.module @DirectDelay(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_delay %1, posedge %0, 2, 0 : !ltl.sequence
  verif.assert %2 label "direct_delay" : !ltl.sequence
  hw.output
}

// CHECK-LABEL: hw.module @RangedDelay(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[CLOCK]] initial
// CHECK:         %[[PAST_A:.+]] = seq.compreg %[[SAMPLED_A]], %[[CLOCK]] initial
// CHECK:         %[[MATCH:.+]] = comb.or %[[PAST_A]], %[[SAMPLED_A]] : i1
// CHECK:         %[[VALID_0:.+]] = seq.compreg %{{.+}}, %[[CLOCK]] initial
// CHECK:         %[[VALID_1:.+]] = seq.compreg %[[VALID_0]], %[[CLOCK]] initial
// CHECK:         %[[VALID_2:.+]] = seq.compreg %[[VALID_1]], %[[CLOCK]] initial
// CHECK:         %[[VALID_PROPERTY:.+]] = comb.and %[[VALID_2]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID_PROPERTY]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[MATCH]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "ranged_delay" : i1
hw.module @RangedDelay(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_delay %1, posedge %0, 1, 1 : !ltl.sequence
  verif.assert %2 label "ranged_delay" : !ltl.sequence
  hw.output
}

// CHECK-LABEL: hw.module @RangedAntecedent(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK-SAME:    in %[[B:[^, ]+]] : i1
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[CLOCK]] initial
// CHECK:         %[[SAMPLED_B:.+]] = seq.compreg %[[B]], %[[CLOCK]] initial
// CHECK:         %[[CONSEQUENT:.+]] = comb.or %[[SAMPLED_B]] : i1
// CHECK:         %[[NOT_A_0:.+]] = comb.xor %[[SAMPLED_A]], %{{.+}} : i1
// CHECK:         %[[OBLIGATION_0:.+]] = comb.or %[[NOT_A_0]], %[[CONSEQUENT]] : i1
// CHECK:         %[[NOT_A_1:.+]] = comb.xor %[[SAMPLED_A]], %{{.+}} : i1
// CHECK:         %[[OBLIGATION_1:.+]] = comb.or %[[NOT_A_1]], %[[CONSEQUENT]] : i1
// CHECK:         %[[PAST_OBLIGATION_0:.+]] = seq.compreg %[[OBLIGATION_0]], %[[CLOCK]] initial
// CHECK:         %[[PROPERTY:.+]] = comb.and %[[PAST_OBLIGATION_0]], %[[OBLIGATION_1]] : i1
// CHECK:         %[[VALID_0:.+]] = seq.compreg %{{.+}}, %[[CLOCK]] initial
// CHECK:         %[[VALID_1:.+]] = seq.compreg %[[VALID_0]], %[[CLOCK]] initial
// CHECK:         %[[VALID_PROPERTY:.+]] = comb.and %[[VALID_1]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID_PROPERTY]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[PROPERTY]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "ranged_antecedent" : i1
hw.module @RangedAntecedent(in %clock : !seq.clock, in %a : i1, in %b : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_atom %b, posedge %0 : i1
  %3 = ltl.clocked_delay %1, posedge %0, 0, 1 : !ltl.sequence
  %4 = ltl.implication %3, %2 : !ltl.sequence, !ltl.sequence
  verif.assert %4 label "ranged_antecedent" : !ltl.property
  hw.output
}

// CHECK-LABEL: hw.module @Concat(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK-SAME:    in %[[B:[^, ]+]] : i1
// CHECK-SAME:    in %[[C:[^, ]+]] : i1
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[CLOCK]] initial
// CHECK:         %[[SAMPLED_B:.+]] = seq.compreg %[[B]], %[[CLOCK]] initial
// CHECK:         %[[PAST_A:.+]] = seq.compreg %[[SAMPLED_A]], %[[CLOCK]] initial
// CHECK:         %[[CONCAT:.+]] = comb.and %[[PAST_A]], %[[SAMPLED_B]] : i1
// CHECK:         %[[SAMPLED_C:.+]] = seq.compreg %[[C]], %[[CLOCK]] initial
// CHECK:         %[[CONSEQUENT:.+]] = comb.or %[[SAMPLED_C]] : i1
// CHECK:         %[[NOT_CONCAT:.+]] = comb.xor %[[CONCAT]], %{{.+}} : i1
// CHECK:         %[[IMPLICATION:.+]] = comb.or %[[NOT_CONCAT]], %[[CONSEQUENT]] : i1
// CHECK:         %[[PROPERTY:.+]] = comb.and %[[IMPLICATION]] : i1
// CHECK:         %[[VALID_0:.+]] = seq.compreg %{{.+}}, %[[CLOCK]] initial
// CHECK:         %[[VALID_1:.+]] = seq.compreg %[[VALID_0]], %[[CLOCK]] initial
// CHECK:         %[[VALID_PROPERTY:.+]] = comb.and %[[VALID_1]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID_PROPERTY]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[PROPERTY]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "concat" : i1
hw.module @Concat(in %clock : !seq.clock, in %a : i1, in %b : i1, in %c : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_atom %b, posedge %0 : i1
  %3 = ltl.clocked_atom %c, posedge %0 : i1
  %4 = ltl.clocked_delay %2, posedge %0, 1, 0 : !ltl.sequence
  %5 = ltl.concat %1, %4 : !ltl.sequence, !ltl.sequence
  %6 = ltl.implication %5, %3 : !ltl.sequence, !ltl.sequence
  verif.assert %6 label "concat" : !ltl.property
  hw.output
}

// CHECK-LABEL: hw.module @PropertyAlignment(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK:         %[[IMPLICATION_1:.+]] = comb.or %{{.+}}, %{{.+}} : i1
// CHECK:         %[[PROPERTY_1:.+]] = comb.and %[[IMPLICATION_1]] : i1
// CHECK:         %[[IMPLICATION_2:.+]] = comb.or %{{.+}}, %{{.+}} : i1
// CHECK:         %[[PROPERTY_2:.+]] = comb.and %[[IMPLICATION_2]] : i1
// CHECK:         %[[ALIGNED_1:.+]] = seq.compreg %[[PROPERTY_1]], %[[CLOCK]] initial
// CHECK:         %[[ALIGNED_PROPERTY:.+]] = comb.and %[[ALIGNED_1]], %[[PROPERTY_2]] : i1
// CHECK:         %[[VALID_0:.+]] = seq.compreg %{{.+}}, %[[CLOCK]] initial
// CHECK:         %[[VALID_1:.+]] = seq.compreg %[[VALID_0]], %[[CLOCK]] initial
// CHECK:         %[[VALID_2:.+]] = seq.compreg %[[VALID_1]], %[[CLOCK]] initial
// CHECK:         %[[VALID_PROPERTY:.+]] = comb.and %[[VALID_2]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID_PROPERTY]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[ALIGNED_PROPERTY]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "property_alignment" : i1
hw.module @PropertyAlignment(in %clock : !seq.clock, in %a : i1, in %b : i1, in %c : i1, in %d : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_atom %b, posedge %0 : i1
  %3 = ltl.clocked_atom %c, posedge %0 : i1
  %4 = ltl.clocked_atom %d, posedge %0 : i1
  %5 = ltl.clocked_delay %2, posedge %0, 1, 0 : !ltl.sequence
  %6 = ltl.implication %1, %5 : !ltl.sequence, !ltl.sequence
  %7 = ltl.clocked_delay %4, posedge %0, 2, 0 : !ltl.sequence
  %8 = ltl.implication %3, %7 : !ltl.sequence, !ltl.sequence
  %9 = ltl.and %6, %8 : !ltl.property, !ltl.property
  verif.assert %9 label "property_alignment" : !ltl.property
  hw.output
}

// CHECK-LABEL: hw.module @PropertyOrNot(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK:         %[[IMPLICATION_1:.+]] = comb.or %{{.+}}, %{{.+}} : i1
// CHECK:         %[[PROPERTY_1:.+]] = comb.and %[[IMPLICATION_1]] : i1
// CHECK:         %[[IMPLICATION_2:.+]] = comb.or %{{.+}}, %{{.+}} : i1
// CHECK:         %[[PROPERTY_2:.+]] = comb.and %[[IMPLICATION_2]] : i1
// CHECK:         %[[ALIGNED_1:.+]] = seq.compreg %[[PROPERTY_1]], %[[CLOCK]] initial
// CHECK:         %[[DISJUNCTION:.+]] = comb.or %[[ALIGNED_1]], %[[PROPERTY_2]] : i1
// CHECK:         %[[NEGATED:.+]] = comb.xor %[[DISJUNCTION]], %{{.+}} : i1
// CHECK:         %[[VALID_0:.+]] = seq.compreg %{{.+}}, %[[CLOCK]] initial
// CHECK:         %[[VALID_1:.+]] = seq.compreg %[[VALID_0]], %[[CLOCK]] initial
// CHECK:         %[[VALID_2:.+]] = seq.compreg %[[VALID_1]], %[[CLOCK]] initial
// CHECK:         %[[VALID_PROPERTY:.+]] = comb.and %[[VALID_2]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID_PROPERTY]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[NEGATED]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "property_or_not" : i1
hw.module @PropertyOrNot(in %clock : !seq.clock, in %a : i1, in %b : i1, in %c : i1, in %d : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_atom %b, posedge %0 : i1
  %3 = ltl.clocked_atom %c, posedge %0 : i1
  %4 = ltl.clocked_atom %d, posedge %0 : i1
  %5 = ltl.clocked_delay %2, posedge %0, 1, 0 : !ltl.sequence
  %6 = ltl.implication %1, %5 : !ltl.sequence, !ltl.sequence
  %7 = ltl.clocked_delay %4, posedge %0, 2, 0 : !ltl.sequence
  %8 = ltl.implication %3, %7 : !ltl.sequence, !ltl.sequence
  %9 = ltl.or %6, %8 : !ltl.property, !ltl.property
  %10 = ltl.not %9 : !ltl.property
  verif.assert %10 label "property_or_not" : !ltl.property
  hw.output
}

// CHECK-LABEL: hw.module @Assume(
// CHECK:         verif.assume %{{.+}} label "assume_temporal" : i1
hw.module @Assume(in %clock : !seq.clock, in %a : i1, in %b : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, posedge %0 : i1
  %2 = ltl.clocked_atom %b, posedge %0 : i1
  %3 = ltl.clocked_delay %2, posedge %0, 1, 0 : !ltl.sequence
  %4 = ltl.implication %1, %3 : !ltl.sequence, !ltl.sequence
  verif.assume %4 label "assume_temporal" : !ltl.property
  hw.output
}

// CHECK-LABEL: hw.module @NegedgeClock(
// CHECK-SAME:    in %[[CLOCK:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK:         %[[CLOCK_I1:.+]] = seq.from_clock %[[CLOCK]]
// CHECK:         %[[NOT_CLOCK:.+]] = comb.xor %[[CLOCK_I1]], %{{.+}} : i1
// CHECK:         %[[NEGEDGE_CLOCK:.+]] = seq.to_clock %[[NOT_CLOCK]]
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[NEGEDGE_CLOCK]] initial
// CHECK:         %[[PROPERTY:.+]] = comb.or %[[SAMPLED_A]] : i1
// CHECK:         %[[VALID_NOT_CLOCK:.+]] = comb.xor %[[CLOCK_I1]], %{{.+}} : i1
// CHECK:         %[[VALID_CLOCK:.+]] = seq.to_clock %[[VALID_NOT_CLOCK]]
// CHECK:         %[[VALID:.+]] = seq.compreg %{{.+}}, %[[VALID_CLOCK]] initial
// CHECK:         %[[VALID_PROPERTY:.+]] = comb.and %[[VALID]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID_PROPERTY]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[PROPERTY]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "negedge" : i1
hw.module @NegedgeClock(in %clock : !seq.clock, in %a : i1) {
  %0 = seq.from_clock %clock
  %1 = ltl.clocked_atom %a, negedge %0 : i1
  verif.assert %1 label "negedge" : !ltl.sequence
  hw.output
}

// CHECK-LABEL: hw.module @MultipleClocks(
// CHECK-SAME:    in %[[CLOCK_0:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[CLOCK_1:[^, ]+]] : !seq.clock
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK-SAME:    in %[[B:[^, ]+]] : i1
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[CLOCK_0]] initial
// CHECK:         %[[SAMPLED_B:.+]] = seq.compreg %[[B]], %[[CLOCK_1]] initial
// CHECK:         %[[CONSEQUENT:.+]] = comb.or %[[SAMPLED_B]] : i1
// CHECK:         %[[PAST_A:.+]] = seq.compreg %[[SAMPLED_A]], %[[CLOCK_1]] initial
// CHECK:         %[[NOT_A:.+]] = comb.xor %[[PAST_A]], %{{.+}} : i1
// CHECK:         %[[IMPLICATION:.+]] = comb.or %[[NOT_A]], %[[CONSEQUENT]] : i1
// CHECK:         %[[PROPERTY:.+]] = comb.and %[[IMPLICATION]] : i1
// CHECK:         %[[VALID_0:.+]] = seq.compreg %{{.+}}, %[[CLOCK_0]] initial
// CHECK:         %[[VALID_1:.+]] = seq.compreg %[[VALID_0]], %[[CLOCK_1]] initial
// CHECK:         %[[VALID_2:.+]] = seq.compreg %{{.+}}, %[[CLOCK_1]] initial
// CHECK:         %[[VALID_3:.+]] = seq.compreg %[[VALID_2]], %[[CLOCK_1]] initial
// CHECK:         %[[VALID:.+]] = comb.and %[[VALID_1]], %[[VALID_3]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[PROPERTY]] : i1
// CHECK:         verif.assert %[[GUARDED]] label "multiple_clocks" : i1
hw.module @MultipleClocks(in %clock0 : !seq.clock, in %clock1 : !seq.clock, in %a : i1, in %b : i1) {
  %0 = seq.from_clock %clock0
  %1 = seq.from_clock %clock1
  %2 = ltl.clocked_atom %a, posedge %0 : i1
  %3 = ltl.clocked_atom %b, posedge %1 : i1
  %4 = ltl.clocked_delay %3, posedge %1, 1, 0 : !ltl.sequence
  %5 = ltl.implication %2, %4 : !ltl.sequence, !ltl.sequence
  verif.assert %5 label "multiple_clocks" : !ltl.property
  hw.output
}

// CHECK-LABEL: hw.module @SameI1Clock(
// CHECK-SAME:    in %[[CLOCK_I1:[^, ]+]] : i1
// CHECK-SAME:    in %[[A:[^, ]+]] : i1
// CHECK-SAME:    in %[[B:[^, ]+]] : i1
// CHECK:         %[[CLOCK_A:.+]] = seq.to_clock %[[CLOCK_I1]]
// CHECK:         %[[SAMPLED_A:.+]] = seq.compreg %[[A]], %[[CLOCK_A]] initial
// CHECK:         %[[CLOCK_B:.+]] = seq.to_clock %[[CLOCK_I1]]
// CHECK:         %[[SAMPLED_B:.+]] = seq.compreg %[[B]], %[[CLOCK_B]] initial
// CHECK:         %[[CONSEQUENT:.+]] = comb.or %[[SAMPLED_B]] : i1
// CHECK:         %[[PAST_CLOCK:.+]] = seq.to_clock %[[CLOCK_I1]]
// CHECK:         %[[PAST_A:.+]] = seq.compreg %[[SAMPLED_A]], %[[PAST_CLOCK]] initial
// CHECK:         %[[NOT_A:.+]] = comb.xor %[[PAST_A]], %{{.+}} : i1
// CHECK:         %[[IMPLICATION:.+]] = comb.or %[[NOT_A]], %[[CONSEQUENT]] : i1
// CHECK:         %[[PROPERTY:.+]] = comb.and %[[IMPLICATION]] : i1
// CHECK:         %[[VALID_CLOCK_0:.+]] = seq.to_clock %[[CLOCK_I1]]
// CHECK:         %[[VALID_0:.+]] = seq.compreg %{{.+}}, %[[VALID_CLOCK_0]] initial
// CHECK:         %[[VALID_CLOCK_1:.+]] = seq.to_clock %[[CLOCK_I1]]
// CHECK:         %[[VALID_1:.+]] = seq.compreg %[[VALID_0]], %[[VALID_CLOCK_1]] initial
// CHECK:         %[[VALID:.+]] = comb.and %[[VALID_1]] : i1
// CHECK:         %[[NOT_VALID:.+]] = comb.xor %[[VALID]], %{{.+}} : i1
// CHECK:         %[[GUARDED:.+]] = comb.or %[[NOT_VALID]], %[[PROPERTY]] : i1
// CHECK:         verif.assert %[[GUARDED]] : i1
hw.module @SameI1Clock(in %clock : i1, in %a : i1, in %b : i1) {
  %0 = ltl.clocked_atom %a, posedge %clock : i1
  %1 = ltl.clocked_atom %b, posedge %clock : i1
  %2 = ltl.clocked_delay %1, posedge %clock, 1, 0 : !ltl.sequence
  %3 = ltl.implication %0, %2 : !ltl.sequence, !ltl.sequence
  verif.assert %3 : !ltl.property
  hw.output
=======
// CHECK-LABEL: hw.module @UnsupportedLTL
// CHECK-SAME: (in [[A:%.+]] : i1, in [[CLK:%.+]] : i1)
// CHECK: [[ATOM:%.+]] = ltl.clocked_atom [[A]], posedge [[CLK]] : i1
// CHECK: [[DELAY:%.+]] = ltl.clocked_delay [[ATOM]], posedge [[CLK]], 1 : !ltl.sequence
// CHECK: verif.assert [[DELAY]] : !ltl.sequence

hw.module @UnsupportedLTL(in %a: i1, in %clk: i1) {
  %atom = ltl.clocked_atom %a, posedge %clk : i1
  %delay = ltl.clocked_delay %atom, posedge %clk, 1 : !ltl.sequence
  verif.assert %delay : !ltl.sequence
>>>>>>> Stashed changes
}
