// RUN: circt-opt %s --pass-pipeline='builtin.module(any(synth-structural-hash))' | FileCheck %s

// CHECK-LABEL: hw.module @test_structural_hash
hw.module @test_structural_hash(in %a: i2, in %b: i2, in %c: i2, out out1: i2,
                                out out2: i2, out out3: i2, out out4: i2, out out7: i2) {
  // CHECK:      %[[VAL0:.+]] = synth.aig.and_inv %a, %b : i2
  // CHECK-NEXT: %[[VAL1:.+]] = synth.aig.and_inv %a, not %b, not %c : i2
  // CHECK-NEXT: hw.output %[[VAL0]], %[[VAL0]], %[[VAL1]], %[[VAL1]], %[[VAL0]]

  // These two operations are equivalent and should be CSE'd
  %0 = synth.aig.and_inv %a, %b : i2
  %1 = synth.aig.and_inv %b, %a : i2

  // These operations have the same inversion patterns and should be CSE'd
  %2 = synth.aig.and_inv %a, not %b, not %c : i2
  %3 = synth.aig.and_inv not %b, %a, not %c : i2

  // Inverted chain that should be CSE'd.
  %6 = synth.aig.and_inv not %b : i2
  %7 = synth.aig.and_inv not %6 : i2
  %8 = synth.aig.and_inv not %7 : i2
  %9 = synth.aig.and_inv not %8, %a : i2

  hw.output %0, %1, %2, %3, %9 : i2, i2, i2, i2, i2
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

// CHECK-LABEL: func.func @test_func
func.func @test_func(%a: i2, %b: i2) -> (i2, i2) {
  // CHECK-NEXT: %[[VAL0:.+]] = synth.aig.and_inv %arg0, %arg1 : i2
  // CHECK-NEXT: return %[[VAL0]], %[[VAL0]]
  %0 = synth.aig.and_inv %a, %b : i2
  %1 = synth.aig.and_inv %b, %a : i2
  return %0, %1 : i2, i2
}

// CHECK-LABEL: hw.module @xor_inv_hash
hw.module @xor_inv_hash(in %a: i1, in %b: i1, in %c: i1, out o0: i1, out o1: i1) {
  // CHECK: %[[X:.+]] = synth.xor_inv %a, not %b, %c : i1
  // CHECK-NEXT: hw.output %[[X]], %[[X]] : i1, i1
  %0 = synth.xor_inv %a, not %b, %c : i1
  %1 = synth.xor_inv %c, %a, not %b : i1
  hw.output %0, %1 : i1, i1
}
