// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

// CHECK: hw.module @Implication(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[C:%.+]] : !ltl.property, in [[CLK:%.+]] : i1)
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[NOT_A:%.+]] = comb.xor [[A]], [[TRUE]] : i1
// CHECK: [[OR:%.+]] = comb.or [[NOT_A]], [[B]] : i1
// CHECK: verif.assert [[OR]] : i1
// CHECK: verif.clocked_assert [[OR]], posedge [[CLK:%.+]] : i1
// CHECK: [[IMP2:%.+]] = ltl.implication [[A]], [[B]] : i1, i1
// CHECK: [[NOT_IMP2:%.+]] = ltl.not [[IMP2]] : !ltl.property
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
  // Or if there are non-i1 operands (and therefore results)
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
  // Or if there are non-i1 operands (and therefore results)
  %or3 = ltl.or %b, %c : i1, !ltl.property
  verif.assert %or3 : !ltl.property
}
