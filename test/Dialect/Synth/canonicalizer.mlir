// RUN: circt-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @And
hw.module @And(in %a: i4, in %b: i4, out o1: i4, out o2: i4,
               out o3: i4, out o4: i4, out o5: i4, out o6: i4, out o7: i4,
               out o8: i4) {
  // CHECK-NEXT: %c-1_i4 = hw.constant -1 : i4
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %c5_i4 = hw.constant 5 : i4
  // CHECK-NEXT: %[[TMP1:.+]] = synth.aig.and_inv %a, %c5_i4 : i4
  // CHECK-NEXT: %[[TMP2:.+]] = synth.aig.and_inv %a, %b : i4
  // CHECK-NEXT: hw.output %c0_i4, %[[TMP1]], %a, %a, %c0_i4, %[[TMP2]], %c0_i4, %c-1_i4 : i4, i4, i4, i4, i4, i4, i4, i4
  %c0 = hw.constant 0 : i4
  %c2 = hw.constant 2 : i4
  %c7 = hw.constant 7 : i4
  %c15 = hw.constant 15 : i4
  
  // a & 0 -> 0
  %0 = synth.aig.and_inv %a, %c0 : i4

  // a & 7 & ~2 -> a & 5
  %1 = synth.aig.and_inv %a, %c7, not %c2 : i4

  // a & 15 -> a
  %2 = synth.aig.and_inv %a, %c15 : i4

  // a & ~0 -> a
  %3 = synth.aig.and_inv %a, not %c0 : i4

  // a & ~15 -> 0
  %4 = synth.aig.and_inv %a, not %c15 : i4

  // a & a & b -> a & b
  %5 = synth.aig.and_inv %a, %a, %b : i4

  // a & ~a & b -> 0
  %6 = synth.aig.and_inv %a, not %a, %b : i4

  // 15 & 15 -> 15
  %7 = synth.aig.and_inv %c15, %c15 : i4

  hw.output %0, %1, %2, %3, %4, %5, %6, %7 : i4, i4, i4, i4, i4, i4, i4, i4
}

// CHECK-LABEL: @DoubleInversion
hw.module @DoubleInversion(in %a: i1, in %b: i1, out o1: i1, out o2: i1, out o3: i1) {
  // CHECK-NEXT: %[[false:.+]] = hw.constant false
  // CHECK-NEXT: %[[TMP1:.+]] = synth.aig.and_inv not %a, %b : i1
  // CHECK-NEXT: hw.output %a, %[[TMP1]], %[[false]] : i1, i1, i1

  // Test flattening of double inversion: and_inv(not(and_inv(not a))) -> a
  %0 = synth.aig.and_inv not %a : i1
  %1 = synth.aig.and_inv not %0 : i1

  // Test flattening with single inverted input: and_inv(and_inv(not a), b) -> and_inv(not a, b)
  %2 = synth.aig.and_inv not %a : i1
  %3 = synth.aig.and_inv %2, %b : i1

  // Test flattening with additional operands: and_inv(not(not a), not a) -> 0
  %4 = synth.aig.and_inv not %a : i1
  %5 = synth.aig.and_inv not %4, not %a : i1
  hw.output %1, %3, %5 : i1, i1, i1
}

// CHECK-LABEL: hw.module @maj_inv_basic
hw.module @maj_inv_basic(in %a : i1, in %b : i1, out o1 : i1, out o2 : i1, out o3 : i1, out o4 : i1, out o5 : i1) {
  // Single operand, not inverted -> replace with operand
  %0 = synth.mig.maj_inv %a : i1

  // Three operands, two same -> replace with the operand
  %1 = synth.mig.maj_inv %a, %a, %b : i1

  // Three operands, two complementary -> replace with the third
  %2 = synth.mig.maj_inv %a, %b, not %b : i1

  // Two operands same but one inverted -> replace with the third
  %3 = synth.mig.maj_inv %a, not %a, %b : i1

  // Two operands same and both inverted -> replace with the same operand with inversion
  // CHECK: %[[RESULT4:.+]] = synth.mig.maj_inv not %a : i1
  %4 = synth.mig.maj_inv not %a, not %a, %b : i1

  // CHECK: hw.output %a, %a, %a, %b, %[[RESULT4]]
  hw.output %0, %1, %2, %3, %4 : i1, i1, i1, i1, i1
}

// CHECK-LABEL: hw.module @maj_inv_constants_canonicalization
hw.module @maj_inv_constants_canonicalization(in %a : i1, out o1 : i1, out o2 : i1, out o3 : i1) {
  // Two constants equal -> replace with constant
  %c = hw.constant 1 : i1
  %0 = synth.mig.maj_inv %c, %c, %a : i1

  // Two constants complementary -> replace with the third
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %1 = synth.mig.maj_inv %c0, %c1, %a : i1

  // Two constants equal with one inverted -> replace with constant
  %2 = synth.mig.maj_inv not %c0, %c1, %a : i1

  // CHECK: hw.output %true, %a, %true : i1, i1, i1
  hw.output %0, %1, %2 : i1, i1, i1
}

// CHECK-LABEL: hw.module @maj_inv_constants_fold
hw.module @maj_inv_constants_fold(out o1 : i2, out o2 : i2) {
  %c1 = hw.constant 1 : i2
  %c2 = hw.constant 2 : i2
  %c3 = hw.constant 3 : i2

  // not(%c1) = 10
  //     %c2  = 10
  // not(%c3) = 00
  // --------------------
  // maj(10, 10, 00) = 10
  // Fold to 2
  %0 = synth.mig.maj_inv not %c1, %c2, not %c3 : i2

  // Fold to 3
  %1 = synth.mig.maj_inv not %c1, not %c2, %c3, %c1, %c2 : i2
  // CHECK: hw.output %c-2_i2, %c-1_i2 : i2, i2
  hw.output %0, %1 : i2, i2
}
