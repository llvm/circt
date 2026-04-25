// RUN: circt-opt %s --convert-synth-to-comb | FileCheck %s

// CHECK-LABEL: @test
hw.module @test(in %a: i32, in %b: i32, in %c: i32, in %d: i32, out out0: i32) {
  // CHECK: %c-1_i32 = hw.constant -1 : i32
  // CHECK: %[[NOT_A:.+]] = comb.xor bin %a, %c-1_i32 : i32
  // CHECK: %[[NOT_C:.+]] = comb.xor bin %c, %c-1_i32 : i32
  // CHECK: %[[RESULT:.+]] = comb.and bin %[[NOT_A]], %b, %[[NOT_C]], %d : i32
  // CHECK: hw.output %[[RESULT]] : i32
  %0 = synth.aig.and_inv not %a, %b, not %c, %d : i32
  hw.output %0 : i32
}

// CHECK-LABEL: @test_choice
hw.module @test_choice(in %a: i32, in %b: i32, in %c: i32, out out0: i32) {
  // CHECK: hw.output %a : i32
  %0 = synth.choice %a, %b, %c : i32
  hw.output %0 : i32
}

// CHECK-LABEL: @test_xor_inv
hw.module @test_xor_inv(in %a: i8, in %b: i8, in %c: i8, out out0: i8) {
  // CHECK: %c-1_i8 = hw.constant -1 : i8
  // CHECK: %[[NOT_B:.+]] = comb.xor bin %b, %c-1_i8 : i8
  // CHECK: %[[RESULT:.+]] = comb.xor bin %a, %[[NOT_B]], %c : i8
  %0 = synth.xor_inv %a, not %b, %c : i8
  hw.output %0 : i8
}

// CHECK-LABEL: @test_dot
hw.module @test_dot(in %x: i8, in %y: i8, in %z: i8, out out0: i8) {
  // CHECK: %c-1_i8 = hw.constant -1 : i8
  // CHECK: %[[NOT_X:.+]] = comb.xor bin %x, %c-1_i8 : i8
  // CHECK: %[[XY:.+]] = comb.and bin %[[NOT_X]], %y : i8
  // CHECK: %[[OR:.+]] = comb.or bin %z, %[[XY]] : i8
  // CHECK: %[[RESULT:.+]] = comb.xor bin %[[NOT_X]], %[[OR]] : i8
  %0 = synth.dot not %x, %y, %z : i8
  hw.output %0 : i8
}
