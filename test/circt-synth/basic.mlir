// RUN: circt-synth %s | FileCheck %s
// RUN: circt-synth %s --until-before aig-lowering | FileCheck %s --check-prefix=AIG
// RUN: circt-synth %s --until-before aig-lowering --convert-to-comb | FileCheck %s --check-prefix=COMB

// AIG-LABEL: @and
// CHECK-LABEL: @and
// COMB-LABEL: @and
hw.module @and(in %a: i2, in %b: i2, in %c: i2, out and: i2) {
  // AIG-NEXT:  %[[AND_INV:.+]] = aig.and_inv %a, %b, %c : i2
  // AIG-NEXT: hw.output %[[AND_INV]] : i2
  // CHECK-DAG: %[[B_1:.+]] = comb.extract %b from 1 : (i2) -> i1
  // CHECK-DAG: %[[C_1:.+]] = comb.extract %c from 1 : (i2) -> i1
  // CHECK-DAG: %[[AND_INV_0:.+]] = aig.and_inv %0, %[[C_1]] : i1
  // CHECK-DAG: %[[B_0:.+]] = comb.extract %b from 0 : (i2) -> i1
  // CHECK-DAG: %[[C_0:.+]] = comb.extract %c from 0 : (i2) -> i1
  // CHECK-DAG: %[[AND_INV_1:.+]] = aig.and_inv %[[B_0]], %[[C_0]] : i1
  // CHECK-DAG: %[[A_1:.+]] = comb.extract %a from 1 : (i2) -> i1
  // CHECK-DAG: %[[AND_INV_2:.+]] = aig.and_inv %[[A_1]], %[[AND_INV_0]] : i1
  // CHECK-DAG: %[[A_0:.+]] = comb.extract %a from 0 : (i2) -> i1
  // CHECK-DAG: %[[AND_INV_3:.+]] = aig.and_inv %[[A_0]], %[[AND_INV_1]] : i1
  // CHECK-DAG: %[[CONCAT:.+]] = comb.concat %[[AND_INV_2]], %[[AND_INV_3]] : i1, i1
  // CHECK-NEXT: hw.output %[[CONCAT]] : i2
  // COMB-NOT: aig.and_inv
  %0 = comb.and %a, %b, %c : i2
  hw.output %0 : i2
}
