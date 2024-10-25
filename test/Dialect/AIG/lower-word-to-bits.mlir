// RUN: circt-opt %s --aig-lower-word-to-bits | FileCheck %s
// CHECK: hw.module @Basic
hw.module @Basic(in %a: i2, in %b: i2, out f: i2) {
  // CHECK-NEXT: %[[RES0:.+]] = comb.extract %a from 0 : (i2) -> i1
  // CHECK-NEXT: %[[RES1:.+]] = comb.extract %b from 0 : (i2) -> i1
  // CHECK-NEXT: %[[RES2:.+]] = aig.and_inv not %[[RES0]], %[[RES1]] : i1
  // CHECK-NEXT: %[[RES3:.+]] = comb.extract %a from 1 : (i2) -> i1
  // CHECK-NEXT: %[[RES4:.+]] = comb.extract %b from 1 : (i2) -> i1
  // CHECK-NEXT: %[[RES5:.+]] = aig.and_inv not %[[RES3]], %[[RES4]] : i1
  // CHECK-NEXT: %[[RES6:.+]] = comb.concat %[[RES2]], %[[RES5]] : i1, i1
  // CHECK-NEXT: hw.output %[[RES6]] : i2
  %0 = aig.and_inv not %a, %b : i2
  hw.output %0 : i2
}
