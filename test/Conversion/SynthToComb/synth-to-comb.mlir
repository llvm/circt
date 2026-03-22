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

// CHECK-LABEL: @test_maj
hw.module @test_maj(in %a: i32, in %b: i32, in %c: i32, out out0: i32) {
  // CHECK: %c-1_i32 = hw.constant -1 : i32
  // CHECK: %[[NOT_B:.+]] = comb.xor bin %b, %c-1_i32 : i32
  // CHECK: %[[AND1:.+]] = comb.and bin %a, %[[NOT_B]] : i32
  // CHECK: %[[AND2:.+]] = comb.and bin %a, %c : i32
  // CHECK: %[[AND3:.+]] = comb.and bin %[[NOT_B]], %c : i32
  // CHECK: %[[RESULT:.+]] = comb.or bin %[[AND1]], %[[AND2]], %[[AND3]] : i32
  %0 = synth.mig.maj_inv %a, not %b, %c : i32
  hw.output %0 : i32
}

// CHECK-LABEL: @test_choice
hw.module @test_choice(in %a: i32, in %b: i32, in %c: i32, out out0: i32) {
  // CHECK: hw.output %a : i32
  %0 = synth.choice %a, %b, %c : i32
  hw.output %0 : i32
}

// CHECK-LABEL: @test_maj5
hw.module @test_maj5(in %a: i32, in %b: i32, in %c: i32, in %d: i32, in %e: i32,
                     out out0: i32) {
  // CHECK: %c-1_i32 = hw.constant -1 : i32
  // CHECK: %[[NOT_B:.+]] = comb.xor bin %b, %c-1_i32 : i32
  // CHECK: %[[ABC:.+]] = comb.and bin %a, %[[NOT_B]], %c : i32
  // CHECK: %[[ABD:.+]] = comb.and bin %a, %[[NOT_B]], %d : i32
  // CHECK: %[[ABE:.+]] = comb.and bin %a, %[[NOT_B]], %e : i32
  // CHECK: %[[ACD:.+]] = comb.and bin %a, %c, %d : i32
  // CHECK: %[[ACE:.+]] = comb.and bin %a, %c, %e : i32
  // CHECK: %[[ADE:.+]] = comb.and bin %a, %d, %e : i32
  // CHECK: %[[BCD:.+]] = comb.and bin %[[NOT_B]], %c, %d : i32
  // CHECK: %[[BCE:.+]] = comb.and bin %[[NOT_B]], %c, %e : i32
  // CHECK: %[[BDE:.+]] = comb.and bin %[[NOT_B]], %d, %e : i32
  // CHECK: %[[CDE:.+]] = comb.and bin %c, %d, %e : i32
  // CHECK: %[[RESULT:.+]] = comb.or bin %[[ABC]], %[[ABD]], %[[ABE]], %[[ACD]], %[[ACE]], %[[ADE]], %[[BCD]], %[[BCE]], %[[BDE]], %[[CDE]] : i32
  // CHECK: hw.output %[[RESULT]] : i32
  %0 = synth.mig.maj_inv %a, not %b, %c, %d, %e : i32
  hw.output %0 : i32
}
