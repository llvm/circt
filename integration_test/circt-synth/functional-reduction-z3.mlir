// REQUIRES: z3-integration
// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(synth-functional-reduction{num-random-patterns=64}))' | FileCheck %s

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
