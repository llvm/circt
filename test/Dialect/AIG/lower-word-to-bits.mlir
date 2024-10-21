// RUN: circt-opt %s --aig-lower-word-to-bits | FileCheck %s
// CHECK: hw.module @Basic
hw.module @Basic(in %a: i2, in %b: i2, out f: i2) {
  // CHECK-NEXT: %0 = comb.extract %a from 0 : (i2) -> i1
  // CHECK-NEXT: %1 = comb.extract %b from 0 : (i2) -> i1
  // CHECK-NEXT: %2 = aig.and_inv not %0, %1 : i1
  // CHECK-NEXT: %3 = comb.extract %a from 1 : (i2) -> i1
  // CHECK-NEXT: %4 = comb.extract %b from 1 : (i2) -> i1
  // CHECK-NEXT: %5 = aig.and_inv not %3, %4 : i1
  // CHECK-NEXT: %6 = comb.concat %2, %5 : i1, i1
  // CHECK-NEXT: hw.output %6 : i2
  %0 = aig.and_inv not %a, %b : i2
  hw.output %0 : i2
}
