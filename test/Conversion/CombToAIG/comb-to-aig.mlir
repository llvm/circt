// RUN: circt-opt %s --convert-comb-to-aig | FileCheck %s

// CHECK-LABEL: @test
hw.module @test(in %arg0: i32, in %arg1: i32, in %arg2: i32, in %arg3: i32, in %arg4: i1, out out: i32) {
  // CHECK-NEXT: %0 = aig.and_inv not %arg0, not %arg1, not %arg2, not %arg3 : i32
  // CHECK-NEXT: %1 = aig.and_inv not %0 : i32
  %0 = comb.or %arg0, %arg1, %arg2, %arg3 : i32
  hw.output %0 : i32
}