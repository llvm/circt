// RUN: circt-opt %s --lower-ltl-to-core | FileCheck %s

// CHECK: hw.module @Implication(in [[A:%.+]] : i1, in [[B:%.+]] : i1, in [[C:%.+]] : !ltl.property)
// CHECK: [[TRUE:%.+]] = hw.constant true
// CHECK: [[NOT_A:%.+]] = comb.xor [[A]], [[TRUE]] : i1
// CHECK: {{.*}} = comb.or [[NOT_A]], [[B]] : i1
// CHECK: {{.*}} = ltl.implication [[B]], [[C]] : i1, !ltl.property

hw.module @Implication(in %a: i1, in %b: i1, in %c: !ltl.property) {
  %imp1 = ltl.implication %a, %b : i1, i1
  %imp2 = ltl.implication %b, %c : i1, !ltl.property
}
