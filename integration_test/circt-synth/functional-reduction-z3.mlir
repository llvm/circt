// REQUIRES: z3-integration, libz3
// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(synth-functional-reduction{num-random-patterns=64 sat-solver=z3}))' -o %t.mlir
// RUN: cat %t.mlir | FileCheck %s

// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 functional_reduction_sat --c2 functional_reduction_sat | FileCheck %s --check-prefix=BASIC
// BASIC: c1 == c2
// SAT should prove that AND(AND(a, not b), AND(c, not d)) is equivalent to
// AND(a, not b, c, not d), and the pass should materialize that with a choice.
// CHECK-LABEL: hw.module @functional_reduction_sat
hw.module @functional_reduction_sat(in %a: i1, in %b: i1, in %c: i1, in %d: i1,
                                    out out1: i1, out out2: i1, out out3: i1) {
  // CHECK: %[[AB:.+]] = synth.aig.and_inv %a, not %b : i1
  // CHECK-NEXT: %[[CD:.+]] = synth.aig.and_inv %c, not %d : i1
  // CHECK-NEXT: %[[TREE:.+]] = synth.aig.and_inv %[[AB]], %[[CD]] : i1
  // CHECK-NEXT: %[[FLAT:.+]] = synth.aig.and_inv %a, not %b, %c, not %d : i1
  // CHECK-NEXT: %[[CHOICE:.+]] = synth.choice %[[TREE]], %[[FLAT]]
  // CHECK-NEXT: %[[DIFF:.+]] = synth.aig.and_inv %a, not %b, not %c, not %d : i1
  // CHECK-NEXT: hw.output %[[CHOICE]], %[[CHOICE]], %[[DIFF]] : i1, i1, i1
  %0 = synth.aig.and_inv %a, not %b : i1
  %1 = synth.aig.and_inv %c, not %d : i1
  %2 = synth.aig.and_inv %0, %1 : i1
  %3 = synth.aig.and_inv %a, not %b, %c, not %d : i1
  %4 = synth.aig.and_inv %a, not %b, not %c, not %d : i1
  hw.output %2, %3, %4 : i1, i1, i1
}

// SAT should also prove equivalence across the newly supported comb and MIG
// nodes.
// RUN: circt-lec %s %t.mlir --shared-libs=%libz3 --c1 functional_reduction_supported_ops_sat --c2 functional_reduction_supported_ops_sat | FileCheck %s --check-prefix=MIXED
// MIXED: c1 == c2
// CHECK-LABEL: hw.module @functional_reduction_supported_ops_sat
hw.module @functional_reduction_supported_ops_sat(
    in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1,
    out out0: i1, out out1: i1, out out2: i1, out out3: i1,
    out out4: i1, out out5: i1, out out6: i1, out out7: i1,
    out out8: i1, out out9: i1) {
  // CHECK: hw.output %[[ORCHOICE:.+]], %[[ORCHOICE]],
  // CHECK-SAME:      %[[ANDCHOICE:.+]], %[[ANDCHOICE]],
  // CHECK-SAME:      %[[XORCHOICE:.+]], %[[XORCHOICE]],
  // CHECK-SAME:      %[[MAJCHOICE:.+]], %[[MAJCHOICE]],
  // CHECK-SAME:      %[[MIG5_CHOICE:.+]], %[[MIG5_CHOICE]]
  %false = hw.constant false
  %0 = comb.or %a, %b, %c : i1
  %1 = comb.or %c, %b, %a : i1
  %2 = comb.and %a, %b : i1
  %3 = synth.mig.maj_inv %a, %b, %false : i1
  %4 = comb.xor %a, %b : i1
  %5 = comb.xor %b, %a : i1
  %6 = synth.mig.maj_inv %a, %b, %c : i1
  %7 = comb.and %c, %4 : i1
  %8 = comb.or %2, %7 : i1
  %9 = synth.mig.maj_inv %a, %b, %c, %d, %e : i1
  %10 = synth.mig.maj_inv %b, %c, %d : i1
  %11 = synth.mig.maj_inv %b, %d, %e : i1
  %12 = synth.mig.maj_inv %b, %c, %e : i1
  %13 = synth.mig.maj_inv %a, %12, %11 : i1
  %14 = synth.mig.maj_inv %a, %10, %13 : i1
  hw.output %0, %1, %2, %3, %4, %5, %6, %8, %9, %14 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
}
