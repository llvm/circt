// RUN: circt-opt %s --aig-lower-variadic | FileCheck %s
// CHECK: hw.module @Basic
hw.module @Basic(in %a: i2, in %b: i2, in %c: i2, in %d: i2, in %e: i2, out f: i2) {
  // CHECK:      %0 = aig.and_inv not %a, %b : i2
  // CHECK-NEXT: %1 = aig.and_inv not %d, %e : i2
  // CHECK-NEXT: %2 = aig.and_inv %c, %1 : i2
  // CHECK-NEXT: %3 = aig.and_inv %0, %2 : i2
  // CHECK-NEXT: hw.output %3 : i2
  %0 = aig.and_inv not %a, %b, %c, not %d, %e : i2
  hw.output %0 : i2
}
