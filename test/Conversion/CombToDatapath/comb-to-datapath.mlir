// RUN: circt-opt %s --convert-comb-to-datapath | FileCheck %s

// CHECK-LABEL: @test
hw.module @test(in %arg0: i4, in %arg1: i4, in %arg2: i4, in %arg3: i4) {
  // CHECK-NEXT: %c42_i32 = hw.constant 42 : i32
  %c42_i32 = hw.constant 42 : i32

  // CHECK-NEXT: comb.add %arg0, %arg1 : i4
  %0 = comb.add %arg0, %arg1 : i4

  // CHECK-NEXT: %[[COMP1:.+]]:2 = datapath.compress %arg0, %arg1, %arg2, %arg3 : i4 [4 -> 2]
  // CHECK-NEXT: comb.add bin %[[COMP1]]#0, %[[COMP1]]#1 : i4
  %1 = comb.add %arg0, %arg1, %arg2, %arg3 : i4
  
  // CHECK-NEXT: %[[PP:.+]]:4 = datapath.partial_product %arg0, %arg1 : (i4, i4) -> (i4, i4, i4, i4)
  // CHECK-NEXT: %[[COMP2:.+]]:2 = datapath.compress %[[PP]]#0, %[[PP]]#1, %[[PP]]#2, %[[PP]]#3 : i4 [4 -> 2]
  // CHECK-NEXT: comb.add bin %[[COMP2]]#0, %[[COMP2]]#1 : i4
  %2 = comb.mul %arg0, %arg1 : i4

  // CHECK-NEXT: comb.mul %arg0, %arg1, %arg2 : i4
  %7 = comb.mul %arg0, %arg1, %arg2 : i4
}

// CHECK-LABEL: @zero_width
hw.module @zero_width(in %arg0: i0, in %arg1: i0, in %arg2: i0) {
  // CHECK-NEXT: hw.constant 0 : i0
  %0 = comb.add %arg0, %arg1, %arg2 : i0
}

// CHECK-LABEL: @carry_in_one
hw.module @carry_in_one(in %arg0: i4, in %arg1: i4) {
  // CHECK-NEXT: %c1_i4 = hw.constant 1 : i4
  %c1_i4 = hw.constant 1 : i4
  // CHECK-NEXT: comb.add %arg0, %arg1, %c1_i4 : i4
  %0 = comb.add %arg0, %arg1, %c1_i4 : i4
}

// CHECK-LABEL: @sub
hw.module @sub(in %arg0: i4, in %arg1: i4, out out: i4) {
  %0 = comb.sub %arg0, %arg1 : i4
  // CHECK-NEXT: %c-1_i4 = hw.constant -1 : i4
  // CHECK-NEXT: %[[XOR:.+]] = comb.xor %arg1, %c-1_i4 : i4
  // CHECK-NEXT: %c1_i4 = hw.constant 1 : i4
  // CHECK-NEXT: %[[ADD:.+]] = comb.add %arg0, %[[XOR]], %c1_i4 : i4
  // CHECK-NEXT: hw.output %[[ADD]] : i4
  hw.output %0 : i4
}

// CHECK-LABEL: func.func @test_no_module
func.func @test_no_module(%arg0: i4, %arg1: i4) -> i4 {
  // CHECK: datapath.partial_product
  // CHECK: datapath.compress
  %0 = comb.mul %arg0, %arg1 : i4
  return %0 :i4
}

