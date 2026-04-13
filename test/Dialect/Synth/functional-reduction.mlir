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

// CHECK-LABEL: hw.module @test_supported_ops
hw.module @test_supported_ops(in %a: i1, in %b: i1, in %c: i1,
                              out out0: i1, out out1: i1,
                              out out2: i1, out out3: i1,
                              out out4: i1, out out5: i1) {
  // CHECK: %[[OR0:.+]] = comb.or %a, %b, %c
  // CHECK: %[[OR1:.+]] = comb.or %c, %b, %a
  // CHECK-NEXT: %[[ORCHOICE:.+]] = synth.choice %[[OR0]], %[[OR1]] : i1
  // CHECK: %[[XOR0:.+]] = comb.xor %a, %b
  // CHECK: %[[XOR1:.+]] = comb.xor %b, %a
  // CHECK-NEXT: %[[XORCHOICE:.+]] = synth.choice %[[XOR0]], %[[XOR1]] : i1
  // CHECK: %[[AND0:.+]] = comb.and %a, %b
  // CHECK: %[[AND1:.+]] = comb.and %b, %a
  // CHECK-NEXT: %[[ANDCHOICE:.+]] = synth.choice %[[AND0]], %[[AND1]] : i1
  // CHECK: hw.output %[[ORCHOICE]], %[[ORCHOICE]], %[[XORCHOICE]], %[[XORCHOICE]], %[[ANDCHOICE]], %[[ANDCHOICE]] : i1, i1, i1, i1, i1, i1
  %0 = comb.or %a, %b, %c {synth.test.fc_equiv_class = 2} : i1
  %1 = comb.or %c, %b, %a {synth.test.fc_equiv_class = 2} : i1
  %2 = comb.xor %a, %b {synth.test.fc_equiv_class = 4} : i1
  %3 = comb.xor %b, %a {synth.test.fc_equiv_class = 4} : i1
  %4 = comb.and %a, %b {synth.test.fc_equiv_class = 5} : i1
  %5 = comb.and %b, %a {synth.test.fc_equiv_class = 5} : i1
  hw.output %0, %1, %2, %3, %4, %5 : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @test_inversion_equiv
hw.module @test_inversion_equiv(in %a: i1, in %b: i1, out out0: i1, out out1: i1) {
  // CHECK: %[[AND:.+]] = synth.aig.and_inv not %a, not %b
  // CHECK: %[[OR:.+]] = comb.or %a, %b
  // CHECK: %[[NOTMEMBER:.+]] = synth.aig.and_inv not %[[OR]]
  // CHECK: %[[CHOICE:.+]] = synth.choice %[[AND]], %[[NOTMEMBER]] : i1
  // CHECK: %[[CHOICENOT:.+]] = synth.aig.and_inv not %[[CHOICE]]
  // CHECK: hw.output %[[CHOICE]], %[[CHOICENOT]]
  %0 = synth.aig.and_inv not %a, not %b {synth.test.fc_equiv_class = 10} : i1
  %1 = comb.or %a, %b {synth.test.fc_equiv_class = 10} : i1
  hw.output %0, %1 : i1, i1
}
