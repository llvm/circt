// RUN: circt-opt %s --synth-functional-reduction=test-transformation | FileCheck %s

// AND(AND(a, not b), AND(c, not d)) is equivalent to AND(a, not b, c, not d).
// CHECK-LABEL: hw.module @test_mixed
hw.module @test_mixed(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out out1: i1, out out2: i1, out out3: i1) {
  // CHECK: %[[RESULT_0:.+]] = synth.aig.and_inv %a, not %b
  // CHECK: %[[RESULT_1:.+]] = synth.aig.and_inv %c, not %d
  // CHECK: %[[RESULT_2:.+]] = synth.aig.and_inv %[[RESULT_0]], %[[RESULT_1]]
  // CHECK: %[[RESULT_3:.+]] = synth.aig.and_inv %a, not %b, %c, not %d
  // CHECK-NEXT: %[[CHOICE:.+]] = synth.choice %[[RESULT_2]], %[[RESULT_3]]
  // CHECK: %[[RESULT_4:.+]] = synth.aig.and_inv %a, not %b, not %c, not %d
  // CHECK-NEXT: hw.output %[[CHOICE]], %[[CHOICE]], %[[RESULT_4]]
  %0 = synth.aig.and_inv %a, not %b : i1
  %1 = synth.aig.and_inv %c, not %d : i1
  %2 = synth.aig.and_inv %0, %1 {synth.test.fc_equiv_class = 0} : i1
  %3 = synth.aig.and_inv %a, not %b, %c, not %d {synth.test.fc_equiv_class = 0} : i1
  %4 = synth.aig.and_inv %a, not %b, not %c, not %d {synth.test.fc_equiv_class = 1} : i1
  hw.output %2, %3, %4 : i1, i1, i1
}
