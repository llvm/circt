// Run cp as a round-trip test.
// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(aig-runner{solver-path="cp" solver-args="<inputFile>" solver-args="<outputFile>"}))' | FileCheck %s

// CHECK-LABEL: hw.module @test(in %a : i1, in %b : i1, out out : i1)
// CHECK-NEXT: %0 = aig.and_inv %b, %a : i1
// CHECK-NEXT: hw.output %0 : i1 
hw.module @test(in %a: i1, in %b: i1, out out: i1) {
  %0 = aig.and_inv %b, %a : i1
  hw.output %0 : i1
}

// Make sure that the unknown operation are properly handled.
// CHECK-LABEL: hw.module @unknownOperation(in %a : i2, in %b : i2, out out : i2) {
// CHECK-NEXT: %[[B_0:.+]] = comb.extract %b from 0 : (i2) -> i1
// CHECK-NEXT: %[[B_1:.+]] = comb.extract %b from 1 : (i2) -> i1
// CHECK-NEXT: %[[A_0:.+]] = comb.extract %a from 0 : (i2) -> i1
// CHECK-NEXT: %[[A_1:.+]] = comb.extract %a from 1 : (i2) -> i1
// CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %[[AND_INV_0:.+]], %[[AND_INV_1:.+]] : i1, i1
// CHECK-NEXT: dbg.variable "test", %[[CONCAT:.+]] : i2
// CHECK-NEXT: %[[A:.+]] = comb.concat %[[A_1]], %[[A_0]] : i1, i1
// CHECK-NEXT: %[[B:.+]] = comb.concat %[[B_1]], %[[B_0]] : i1, i1
// CHECK-NEXT: %[[ADD:.+]] = comb.add %[[A]], %[[B]] : i2
// CHECK-NEXT: %[[ADD_0:.+]] = comb.extract %[[ADD]] from 0
// CHECK-NEXT: %[[ADD_1:.+]] = comb.extract %[[ADD]] from 1
// CHECK-NEXT: %[[CONCAT:.+]] = comb.concat %[[AND_INV_4:.+]], %[[AND_INV_3:.+]] : i1, i1
// CHECK-NEXT: %[[AND_INV_0:.+]] = aig.and_inv %[[B_0]], %[[A_0]] : i1
// CHECK-NEXT: %[[AND_INV_1:.+]] = aig.and_inv %[[B_1]], %[[A_1]] : i1
// CHECK-NEXT: %[[AND_INV_2:.+]] = aig.and_inv %[[ADD_0]], %[[A_0]] : i1
// CHECK-NEXT: %[[AND_INV_3:.+]] = aig.and_inv %[[ADD_1]], %[[A_1]] : i1
// CHECK-NEXT: hw.output %[[CONCAT:.+]] : i2
hw.module @unknownOperation(in %a: i2, in %b: i2, out out: i2) {
  %0 = aig.and_inv %b, %a : i2
  dbg.variable "test", %0 : i2
  %1 = comb.add %a, %b : i2
  %2 = aig.and_inv %1, %a : i2
  hw.output %2 : i2
}
