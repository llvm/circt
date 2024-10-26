// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: @And
// CHECK-NEXT: %[[RES0:.+]] = aig.and_inv %b, %b : i4
// CHECK-NEXT: %[[RES1:.+]] = aig.and_inv %b, not %b : i4
// CHECK-NEXT: %[[RES2:.+]] = aig.and_inv not %a, not %a : i1
hw.module @And(in %a: i1, in %b: i4) {
  %0 = aig.and_inv %b, %b : i4
  %1 = aig.and_inv %b, not %b : i4
  %2 = aig.and_inv not %a, not %a : i1
}
