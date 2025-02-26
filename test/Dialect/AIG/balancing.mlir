// RUN: circt-opt %s --maximum-and-cover --aig-balance-variadic | FileCheck %s
// CHECK-LABEL: @Tree1
hw.module @Tree1(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1, in %g: i1, out o1: i1) {
  // CHECK-NEXT: %[[AND_INV0:.+]] = aig.and_inv %d, not %e : i1
  // CHECK-NEXT: %[[AND_INV1:.+]] = aig.and_inv not %c, %f : i1
  // CHECK-NEXT: %[[AND_INV2:.+]] = aig.and_inv not %[[AND_INV0]], %[[AND_INV1]] : i1
  // CHECK-NEXT: %[[AND_INV3:.+]] = aig.and_inv %a, not %b : i1
  // CHECK-NEXT: %[[AND_INV4:.+]] = aig.and_inv %g, %[[AND_INV3]] : i1
  // CHECK-NEXT: %[[AND_INV5:.+]] = aig.and_inv not %[[AND_INV2]], %[[AND_INV4]] : i1
  // CHECK-NEXT: hw.output %[[AND_INV5]] : i1
  
  %1 = aig.and_inv %a, not %b : i1
  %2 = aig.and_inv %d, not %e : i1
  %3 = aig.and_inv not %2, %f : i1
  %4 = aig.and_inv not %c, %3 : i1
  %5 = aig.and_inv %1, not %4 : i1
  %6 = aig.and_inv %5, %g : i1

  hw.output %6 : i1
}

// CHECK-LABEL: @Tree2
hw.module @Tree2(in %a: i1, in %b: i1, in %c: i1, in %d: i1, out o1: i1) {
  // CHECK-NEXT: %[[AND_INV0:.+]] = aig.and_inv %a, %b : i1
  // CHECK-NEXT: %[[AND_INV1:.+]] = aig.and_inv %c, %d : i1
  // CHECK-NEXT: %[[AND_INV2:.+]] = aig.and_inv %[[AND_INV0]], %[[AND_INV1]] : i1
  // CHECK-NEXT: hw.output %[[AND_INV2]] : i1

  %1 = aig.and_inv %a, %b : i1
  %2 = aig.and_inv %c, %1 : i1
  %3 = aig.and_inv %d, %2 : i1

  hw.output %3 : i1
}
