// RUN: circt-opt %s --maximum-and-cover --aig-balance-variadic | FileCheck %s
// CHECK-LABEL: @Tree1
hw.module @Tree1(in %a: i1, in %b: i1, in %c: i1, in %d: i1, in %e: i1, in %f: i1, in %g: i1, out o1: i1) {
 // CHECK-NEXT: %0 = aig.and_inv %d, not %e : i1
 // CHECK-NEXT: %1 = aig.and_inv not %c, %f : i1
 // CHECK-NEXT: %2 = aig.and_inv not %0, %1 : i1
 // CHECK-NEXT: %3 = aig.and_inv %a, not %b : i1
 // CHECK-NEXT: %4 = aig.and_inv %g, %3 : i1
 // CHECK-NEXT: %5 = aig.and_inv not %2, %4 : i1
 // CHECK-NEXT: hw.output %5 : i1
  
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
  // CHECK-NEXT: %0 = aig.and_inv %d, %c : i1
  // CHECK-NEXT: %1 = aig.and_inv %b, %a : i1
  // CHECK-NEXT: %2 = aig.and_inv %0, %1 : i1

  %1 = aig.and_inv %a, %b : i1
  %2 = aig.and_inv %c, %1 : i1
  %3 = aig.and_inv %d, %2 : i1

  hw.output %3 : i1
}
