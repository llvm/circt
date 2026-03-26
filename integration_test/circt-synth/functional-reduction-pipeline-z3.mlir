// REQUIRES: libz3
// RUN: circt-synth %s --until-before mapping | FileCheck %s --check-prefix=DISABLED
// RUN: circt-synth %s --enable-functional-reduction --until-before mapping | FileCheck %s --check-prefix=ENABLED

// `circt-synth` keeps FunctionalReduction opt-in. Without the flag, the two
// equivalent majority implementations stay distinct through synthesis.
// DISABLED-LABEL: hw.module @maj_equiv
// DISABLED-NOT: synth.choice
// DISABLED: hw.output %[[LEFT:.+]], %[[RIGHT:.+]] : i1, i1

// With the flag enabled, FunctionalReduction proves the two AIG networks
// equivalent and materializes the merge using choices.
// ENABLED-LABEL: hw.module @maj_equiv
// ENABLED: %[[CHOICE0:.+]] = synth.choice %{{.+}}, %{{.+}} : i1
// ENABLED: %[[CHOICE1:.+]] = synth.choice %{{.+}}, %{{.+}} : i1
// ENABLED: hw.output %[[CHOICE1]], %[[CHOICE1]] : i1, i1
hw.module @maj_equiv(in %a: i1, in %b: i1, in %c: i1, out out1: i1, out out2: i1) {
  %ab0 = comb.and %a, %b : i1
  %ac0 = comb.and %a, %c : i1
  %bc0 = comb.and %b, %c : i1
  %lhs = comb.or %ab0, %ac0, %bc0 : i1

  %ab1 = comb.and %a, %b : i1
  %xor = comb.xor %a, %b : i1
  %cxor = comb.and %c, %xor : i1
  %rhs = comb.or %ab1, %cxor : i1
  hw.output %lhs, %rhs : i1, i1
}
