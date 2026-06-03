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

// CHECK-LABEL: @test_majority
hw.module @test_majority(in %a: i8, in %b: i8, in %c: i8, out out0: i8) {
  // CHECK: %[[AB:.+]] = comb.and bin %a, %b : i8
  // CHECK: %[[AC:.+]] = comb.and bin %a, %c : i8
  // CHECK: %[[BC:.+]] = comb.and bin %b, %c : i8
  // CHECK: %[[OR:.+]] = comb.or bin %[[AB]], %[[AC]] : i8
  // CHECK: %[[RESULT:.+]] = comb.or bin %[[OR]], %[[BC]] : i8
  %0 = synth.majority %a, %b, %c : i8
  hw.output %0 : i8
}

// CHECK-LABEL: @test_onehot
hw.module @test_onehot(in %a: i8, in %b: i8, in %c: i8, out out0: i8) {
  // CHECK: %c-1_i8 = hw.constant -1 : i8
  // CHECK: %[[NOT_A:.+]] = comb.xor bin %a, %c-1_i8 : i8
  // CHECK: %[[NOT_B:.+]] = comb.xor bin %b, %c-1_i8 : i8
  // CHECK: %[[NOT_C:.+]] = comb.xor bin %c, %c-1_i8 : i8
  // CHECK: %[[A_ONLY:.+]] = comb.and bin %a, %[[NOT_B]], %[[NOT_C]] : i8
  // CHECK: %[[B_ONLY:.+]] = comb.and bin %[[NOT_A]], %b, %[[NOT_C]] : i8
  // CHECK: %[[C_ONLY:.+]] = comb.and bin %[[NOT_A]], %[[NOT_B]], %c : i8
  // CHECK: %[[RESULT:.+]] = comb.or bin %[[A_ONLY]], %[[B_ONLY]], %[[C_ONLY]] : i8
  %0 = synth.onehot %a, %b, %c : i8
  hw.output %0 : i8
}

// CHECK-LABEL: @test_mux_inv
hw.module @test_mux_inv(in %c: i8, in %a: i8, in %b: i8, out out0: i8) {
  // CHECK: %c-1_i8 = hw.constant -1 : i8
  // CHECK: %[[NOT_C:.+]] = comb.xor bin %c, %c-1_i8 : i8
  // CHECK: %[[NOT_B:.+]] = comb.xor bin %b, %c-1_i8 : i8
  // CHECK: %[[TRUE:.+]] = comb.and bin %[[NOT_C]], %a : i8
  // CHECK: %[[FALSE:.+]] = comb.and bin %c, %[[NOT_B]] : i8
  // CHECK: %[[RESULT:.+]] = comb.or bin %[[TRUE]], %[[FALSE]] : i8
  %0 = synth.mux_inv not %c, %a, not %b : i8
  hw.output %0 : i8
}

// CHECK-LABEL: @test_gamble
hw.module @test_gamble(in %a: i8, in %b: i8, in %c: i8, out out0: i8) {
  // CHECK: %c-1_i8 = hw.constant -1 : i8
  // CHECK: %[[ALL_SET:.+]] = comb.and bin %a, %b, %c : i8
  // CHECK: %[[OR_SET:.+]] = comb.or bin %a, %b, %c : i8
  // CHECK: %[[NONE_SET:.+]] = comb.xor bin %[[OR_SET]], %c-1_i8 : i8
  // CHECK: %[[RESULT:.+]] = comb.or bin %[[ALL_SET]], %[[NONE_SET]] : i8
  %0 = synth.gamble %a, %b, %c : i8
  hw.output %0 : i8
}
