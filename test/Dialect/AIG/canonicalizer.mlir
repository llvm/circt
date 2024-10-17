// RUN: circt-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @And
hw.module @And(in %a: i4, in %b: i4, out o1: i4, out o2: i4,
               out o3: i4, out o4: i4, out o5: i4, out o6: i4, out o7: i4,
               out o8: i4) {
  // CHECK-NEXT: %c0_i4 = hw.constant 0 : i4
  // CHECK-NEXT: %c-1_i4 = hw.constant -1 : i4
  // CHECK-NEXT: %c5_i4 = hw.constant 5 : i4
  // CHECK-NEXT: %[[TMP1:.+]] = aig.and_inv %a, %c5_i4 : i4
  // CHECK-NEXT: %[[TMP2:.+]] = aig.and_inv %a, %b : i4
  // CHECK-NEXT: hw.output %c0_i4, %[[TMP1]], %a, %a, %c0_i4, %[[TMP2]], %c0_i4, %c-1_i4 : i4, i4, i4, i4, i4, i4, i4, i4
  %c0 = hw.constant 0 : i4
  %c2 = hw.constant 2 : i4
  %c8 = hw.constant 7 : i4
  %c15 = hw.constant 15 : i4
  
  // a & 0 -> 0
  %0 = aig.and_inv %a, %c0 : i4

  // a & 7 & ~2 -> a & 5
  %1 = aig.and_inv %a, %c8, not %c2 : i4

  // a & 15 -> a
  %2 = aig.and_inv %a, %c15 : i4

  // a & ~0 -> a
  %3 = aig.and_inv %a, not %c0 : i4

  // a & ~15 -> 0
  %4 = aig.and_inv %a, not %c15 : i4

  // a & a & b -> a & b
  %5 = aig.and_inv %a, %b : i4

  // a & ~a & b -> 0
  %6 = hw.constant 0 : i4

  // 15 & 15 -> 15
  %7 = hw.constant -1 : i4
  hw.output %0, %1, %2, %3, %4, %5, %6, %7 : i4, i4, i4, i4, i4, i4, i4, i4
}
