// RUN: circt-opt %s --pass-pipeline='builtin.module(hw.module(convert-comb-to-aig{target-ir=mig},cse))' | FileCheck %s

hw.module @logic(in %arg0: i32, in %arg1: i32, in %arg2: i32, in %arg3: i32, out out0: i32, out out1: i32) {
  // CHECK: %[[MIG_0:.+]] = synth.mig.maj_inv %arg1, %arg2, %{{c-1_i32.*}} : i32
  // CHECK: %[[MIG_1:.+]] = synth.mig.maj_inv %arg0, %[[MIG_0]], %{{c-1_i32.*}} : i32
  // CHECK: %[[MIG_2:.+]] = synth.mig.maj_inv %arg1, %arg2, %{{c0_i32.*}} : i32
  // CHECK: %[[MIG_3:.+]] = synth.mig.maj_inv %arg0, %[[MIG_2]], %{{c0_i32.*}} : i32
  // CHECK: hw.output %[[MIG_1]], %[[MIG_3]] : i32, i32
  %0 = comb.or %arg0, %arg1, %arg2 : i32
  %1 = comb.and %arg0, %arg1, %arg2 : i32
  hw.output %0, %1 : i32, i32
}


// CHECK-LABEL: @add
hw.module @add(in %lhs: i3, in %rhs: i3, out out: i3) {
  // Check majority function is used for carry logic
  // CHECK: %[[LHS_0:.+]] = comb.extract %lhs from 0 : (i3) -> i1
  // CHECK: %[[LHS_1:.+]] = comb.extract %lhs from 1 : (i3) -> i1
  // CHECK: %[[RHS_0:.+]] = comb.extract %rhs from 0 : (i3) -> i1
  // CHECK: %[[RHS_1:.+]] = comb.extract %rhs from 1 : (i3) -> i1
  // CHECK: %[[LHS_0_AND_RHS_0:.+]] = synth.mig.maj_inv %[[LHS_0]], %[[RHS_0]], %{{false.*}} : i1
  // CHECK: %[[CARRY_1:.+]] = synth.mig.maj_inv %[[LHS_1]], %[[RHS_1]], %[[LHS_0_AND_RHS_0]] : i1
  %0 = comb.add %lhs, %rhs : i3
  hw.output %0 : i3
}
