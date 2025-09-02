// RUN: circt-opt %s --aig-lower-word-to-bits | FileCheck %s
// CHECK: hw.module @Basic
hw.module @Basic(in %a: i2, in %b: i2, out f: i2) {
  %0 = aig.and_inv not %a, %b : i2
  %1 = aig.and_inv not %0, not %0 : i2
  // CHECK-NEXT: %[[EXTRACT_A_1:.+]] = comb.extract %a from 1
  // CHECK-NEXT: %[[EXTRACT_B_1:.+]] = comb.extract %b from 1
  // CHECK-NEXT: %[[AND_INV_0:.+]] = aig.and_inv not %[[EXTRACT_A_1]], %[[EXTRACT_B_1]]
  // CHECK-NEXT: %[[EXTRACT_A_0:.+]] = comb.extract %a from 0
  // CHECK-NEXT: %[[EXTRACT_B_0:.+]] = comb.extract %b from 0
  // CHECK-NEXT: %[[AND_INV_1:.+]] = aig.and_inv not %[[EXTRACT_A_0]], %[[EXTRACT_B_0]]
  // CHECK-NEXT: %[[AND_INV_2:.+]] = aig.and_inv not %[[AND_INV_0]], not %[[AND_INV_0]]
  // CHECK-NEXT: %[[AND_INV_3:.+]] = aig.and_inv not %[[AND_INV_1]], not %[[AND_INV_1]]
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %[[AND_INV_2]], %[[AND_INV_3]]
  hw.output %1 : i2
}

// CHECK-LABEL: hw.module @Basic_MIG
hw.module @Basic_MIG(in %a: i2, in %b: i2, in %c: i2, out f: i2) {
  %0 = synth.mig.maj_inv not %a, %b, %c : i2
  // CHECK-NEXT: %[[EXTRACT_A_1:.+]] = comb.extract %a from 1 : (i2) -> i1
  // CHECK-NEXT: %[[EXTRACT_B_1:.+]] = comb.extract %b from 1 : (i2) -> i1
  // CHECK-NEXT: %[[EXTRACT_C_1:.+]] = comb.extract %c from 1 : (i2) -> i1
  // CHECK-NEXT: %[[MAJ_INV_0:.+]] = synth.mig.maj_inv not %[[EXTRACT_A_1]], %[[EXTRACT_B_1]], %[[EXTRACT_C_1]] : i1
  // CHECK-NEXT: %[[EXTRACT_A_0:.+]] = comb.extract %a from 0 : (i2) -> i1
  // CHECK-NEXT: %[[EXTRACT_B_0:.+]] = comb.extract %b from 0 : (i2) -> i1
  // CHECK-NEXT: %[[EXTRACT_C_0:.+]] = comb.extract %c from 0 : (i2) -> i1
  // CHECK-NEXT: %[[MAJ_INV_1:.+]] = synth.mig.maj_inv not %[[EXTRACT_A_0]], %[[EXTRACT_B_0]], %[[EXTRACT_C_0]] : i1
  // CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %[[MAJ_INV_0]], %[[MAJ_INV_1]] : i1, i1
  // CHECK-NEXT: hw.output %[[CONCAT]] : i2
  hw.output %0 : i2
}
