// RUN: circt-opt %s --synth-structural-hash | FileCheck %s

// CHECK-LABEL: hw.module @test_structural_hash
hw.module @test_structural_hash(in %a: i2, in %b: i2, in %c: i2, out out1: i2,
                                out out2: i2, out out3: i2, out out4: i2, out out5: i2, out out6: i2, out out7: i2) {
  // CHECK:      %[[VAL0:.+]] = synth.aig.and_inv %a, %b : i2
  // CHECK-NEXT: %[[VAL1:.+]] = synth.aig.and_inv %a, not %b, not %c : i2
  // CHECK-NEXT: %[[VAL2:.+]] = synth.mig.maj_inv %a, not %b, not %c : i2
  // CHECK-NEXT: hw.output %[[VAL0]], %[[VAL0]], %[[VAL1]], %[[VAL1]], %[[VAL2]], %[[VAL2]], %[[VAL0]]

  // These two operations are equivalent and should be CSE'd
  %0 = synth.aig.and_inv %a, %b : i2
  %1 = synth.aig.and_inv %b, %a : i2

  // These operations have the same inversion patterns and should be CSE'd
  %2 = synth.aig.and_inv %a, not %b, not %c : i2
  %3 = synth.aig.and_inv not %b, %a, not %c : i2

  // The same applies to maj_inv operations
  %4 = synth.mig.maj_inv %a, not %b, not %c : i2
  %5 = synth.mig.maj_inv not %c, not %b, %a : i2

  // Inverted chain that should be CSE'd (regardless of and_inv/maj_inv)
  %6 = synth.aig.and_inv not %b : i2
  %7 = synth.mig.maj_inv not %6 : i2
  %8 = synth.mig.maj_inv not %7 : i2
  %9 = synth.aig.and_inv not %8, %a : i2

  hw.output %0, %1, %2, %3, %4, %5, %9 : i2, i2, i2, i2, i2, i2, i2
}

hw.module.extern @cycle(in %b: i2, out out1: i2)
// CHECK-LABEL: hw.module @topo_sort
hw.module @topo_sort(in %a: i2, in %b: i2, out out1: i2, out out2: i2, out out3: i2) {
  // CHECK:      %[[VAL0:.+]] = hw.instance "cycle" @cycle(b: %b: i2) -> (out1: i2)
  // CHECK-NEXT: %[[AND_INV0:.+]] = synth.aig.and_inv not %a, %[[VAL0]]
  // CHECK-NEXT: %[[AND_INV1:.+]] = synth.aig.and_inv %[[VAL0]], not %[[AND_INV0]]
  // CHECK-NEXT: hw.output %[[AND_INV0]], %[[AND_INV1]], %[[AND_INV1]]
  %2 = synth.aig.and_inv not %0, %c : i2
  %1 = synth.aig.and_inv %c, not %0 : i2
  %0 = synth.aig.and_inv %c, not %a : i2
  %c = hw.instance "cycle" @cycle(b: %b: i2) -> (out1: i2)
  hw.output %0, %1, %2 : i2, i2, i2
}

// CHECK-LABEL: hw.module @port_removal
hw.module @port_removal(in %a: i2) {
  // CHECK-NEXT: hw.output
  %0 = synth.aig.and_inv not %a : i2
}
