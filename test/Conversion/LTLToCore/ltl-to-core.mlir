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
