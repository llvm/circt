// RUN: circt-opt %s --convert-aig-to-comb | FileCheck %s

// CHECK-LABEL: @test
hw.module @test(in %a: i32, in %b: i32, in %c: i32, in %d: i32, out out0: i32) {
  // CHECK: %c-1_i32 = hw.constant -1 : i32
  // CHECK: %[[NOT_A:.+]] = comb.xor %a, %c-1_i32 : i32
  // CHECK: %[[NOT_C:.+]] = comb.xor %c, %c-1_i32 : i32
  // CHECK: %[[RESULT:.+]] = comb.and %[[NOT_A]], %b, %[[NOT_C]], %d : i32
  // CHECK: hw.output %[[RESULT]] : i32
  %0 = aig.and_inv not %a, %b, not %c, %d : i32
  hw.output %0 : i32
}
