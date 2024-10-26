// RUN: circt-opt %s --convert-comb-to-aig | FileCheck %s

// CHECK-LABEL: @test
hw.module @test(in %arg0: i32, in %arg1: i32, in %arg2: i32, in %arg3: i32, out out0: i32, out out1: i32, out out2: i32) {
  // CHECK-NEXT: %[[OR_TMP:.+]] = aig.and_inv not %arg0, not %arg1, not %arg2, not %arg3 : i32
  // CHECK-NEXT: %[[OR:.+]] = aig.and_inv not %0 : i32
  // CHECK-NEXT: %[[AND:.+]] = aig.and_inv %arg0, %arg1, %arg2, %arg3 : i32
  // CHECK-NEXT: %[[XOR_NOT_AND:.+]] = aig.and_inv not %arg0, not %arg1 : i32
  // CHECK-NEXT: %[[XOR_AND:.+]] = aig.and_inv %arg0, %arg1 : i32
  // CHECK-NEXT: %[[XOR:.+]] = aig.and_inv not %[[XOR_NOT_AND]], not %[[XOR_AND]] : i32
  // CHECK-NEXT: hw.output %[[OR]], %[[AND]], %[[XOR]] : i32, i32, i32
  %0 = comb.or %arg0, %arg1, %arg2, %arg3 : i32
  %1 = comb.and %arg0, %arg1, %arg2, %arg3 : i32
  %2 = comb.xor %arg0, %arg1 : i32
  hw.output %0, %1, %2 : i32, i32, i32
}
