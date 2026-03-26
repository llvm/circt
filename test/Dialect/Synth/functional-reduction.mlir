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
                              out out4: i1, out out5: i1,
                              out out6: i1, out out7: i1) {
  // CHECK: %[[FALSE:.+]] = hw.constant false
  // CHECK: %[[OR0:.+]] = comb.or %a, %b, %c
  // CHECK: %[[OR1:.+]] = comb.or %c, %b, %a
  // CHECK-NEXT: %[[ORCHOICE:.+]] = synth.choice %[[OR0]], %[[OR1]] : i1
  // CHECK: %[[AND0:.+]] = comb.and %a, %b
  // CHECK: %[[AND1:.+]] = synth.mig.maj_inv %a, %b, %[[FALSE]]
  // CHECK-NEXT: %[[ANDCHOICE:.+]] = synth.choice %[[AND0]], %[[AND1]] : i1
  // CHECK: %[[XOR0:.+]] = comb.xor %a, %b
  // CHECK: %[[XOR1:.+]] = comb.xor %b, %a
  // CHECK-NEXT: %[[XORCHOICE:.+]] = synth.choice %[[XOR0]], %[[XOR1]] : i1
  // CHECK: %[[MAJ0:.+]] = synth.mig.maj_inv %a, %b, %c
  // CHECK: %[[CXOR:.+]] = comb.and %c, %[[XORCHOICE]]
  // CHECK: %[[MAJ1:.+]] = comb.or %[[ANDCHOICE]], %[[CXOR]]
  // CHECK-NEXT: %[[MAJCHOICE:.+]] = synth.choice %[[MAJ0]], %[[MAJ1]] : i1
  // CHECK: hw.output %[[ORCHOICE]], %[[ORCHOICE]], %[[ANDCHOICE]], %[[ANDCHOICE]], %[[XORCHOICE]], %[[XORCHOICE]], %[[MAJCHOICE]], %[[MAJCHOICE]] : i1, i1, i1, i1, i1, i1, i1, i1
  %false = hw.constant false
  %0 = comb.or %a, %b, %c {synth.test.fc_equiv_class = 2} : i1
  %1 = comb.or %c, %b, %a {synth.test.fc_equiv_class = 2} : i1
  %2 = comb.and %a, %b {synth.test.fc_equiv_class = 3} : i1
  %3 = synth.mig.maj_inv %a, %b, %false {synth.test.fc_equiv_class = 3} : i1
  %4 = comb.xor %a, %b {synth.test.fc_equiv_class = 4} : i1
  %5 = comb.xor %b, %a {synth.test.fc_equiv_class = 4} : i1
  %6 = synth.mig.maj_inv %a, %b, %c {synth.test.fc_equiv_class = 5} : i1
  %7 = comb.and %c, %4 : i1
  %8 = comb.or %2, %7 {synth.test.fc_equiv_class = 5} : i1
  hw.output %0, %1, %2, %3, %4, %5, %6, %8 : i1, i1, i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @test_five_input_mig
hw.module @test_five_input_mig(in %a: i1, in %b: i1, in %c: i1, in %d: i1,
                               in %e: i1, out out0: i1, out out1: i1) {
  // CHECK: %[[M0:.+]] = synth.mig.maj_inv %a, %b, %c, %d, %e
  // CHECK: %[[M1:.+]] = synth.mig.maj_inv %e, %d, %c, %b, %a
  // CHECK-NEXT: %[[CHOICE:.+]] = synth.choice %[[M0]], %[[M1]] : i1
  // CHECK: hw.output %[[CHOICE]], %[[CHOICE]] : i1, i1
  %0 = synth.mig.maj_inv %a, %b, %c, %d, %e {synth.test.fc_equiv_class = 6} : i1
  %1 = synth.mig.maj_inv %e, %d, %c, %b, %a {synth.test.fc_equiv_class = 6} : i1
  hw.output %0, %1 : i1, i1
}
