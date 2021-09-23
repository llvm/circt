// RUN: circt-opt %s -test-dependence-analysis | FileCheck %s

// CHECK-LABEL: func @test1
func @test1(%arg0: memref<?xi32>) -> i32 {
  %c0_i32 = constant 0 : i32
  %0:2 = affine.for %arg1 = 0 to 10 iter_args(%arg2 = %c0_i32, %arg3 = %c0_i32) -> (i32, i32) {
    // CHECK: affine.load %arg0[%arg1] {dependences = []}
    %1 = affine.load %arg0[%arg1] : memref<?xi32>
    %2 = addi %arg2, %1 : i32
    affine.yield %2, %2 : i32, i32
  }
  return %0#1 : i32
}

// CHECK-LABEL: func @test2
#set = affine_set<(d0) : (d0 - 3 >= 0)>
func @test2(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
  affine.for %arg2 = 0 to 10 {
    // CHECK: affine.load %arg0[%arg2] {dependences = []}
    %0 = affine.load %arg0[%arg2] : memref<?xi32>
    affine.if #set(%arg2) {
      // CHECK{LITERAL}: affine.load %arg0[%arg2 - 3] {dependences = [[[3, 3]]]}
      %1 = affine.load %arg0[%arg2 - 3] : memref<?xi32>
      %2 = addi %0, %1 : i32
      // CHECK: affine.store %2, %arg1[%arg2 - 3] {dependences = []}
      affine.store %2, %arg1[%arg2 - 3] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func @test3
func @test3(%arg0: memref<?xi32>) {
  %0 = memref.alloca() : memref<1xi32>
  %1 = memref.alloca() : memref<1xi32>
  %2 = memref.alloca() : memref<1xi32>
  affine.for %arg1 = 0 to 10 {
    // CHECK{LITERAL}: %3 = affine.load %2[0] {dependences = [[[1, 9]]]}
    %3 = affine.load %2[0] : memref<1xi32>
    // CHECK{LITERAL}: %4 = affine.load %1[0] {dependences = [[[1, 9]]]}
    %4 = affine.load %1[0] : memref<1xi32>
    // CHECK{LITERAL}: affine.store %4, %2[0] {dependences = [[[1, 9]]]}
    affine.store %4, %2[0] : memref<1xi32>
    // CHECK{LITERAL}: %5 = affine.load %0[0] {dependences = [[[1, 9]]]}
    %5 = affine.load %0[0] : memref<1xi32>
    // CHECK{LITERAL}: affine.store %5, %1[0] {dependences = [[[1, 9]]]}
    affine.store %5, %1[0] : memref<1xi32>
    // CHECK: affine.load %arg0[%arg1] {dependences = []}
    %6 = affine.load %arg0[%arg1] : memref<?xi32>
    %7 = addi %3, %6 : i32
    // CHECK{LITERAL}: affine.store %7, %0[0] {dependences = [[[1, 9]]]}
    affine.store %7, %0[0] : memref<1xi32>
  }
  return
}

// CHECK-LABEL: func @test4
func @test4(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
  %c1_i32 = constant 1 : i32
  affine.for %arg2 = 0 to 10 {
    // CHECK: affine.load %arg1[%arg2] {dependences = []}
    %0 = affine.load %arg1[%arg2] : memref<?xi32>
    %1 = index_cast %0 : i32 to index
    %2 = memref.load %arg0[%1] : memref<?xi32>
    %3 = addi %2, %c1_i32 : i32
    memref.store %3, %arg0[%1] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func @test5
func @test5(%arg0: memref<?xi32>) {
  affine.for %arg1 = 2 to 10 {
    // CHECK{LITERAL}: affine.load %arg0[%arg1 - 2] {dependences = [[[1, 1]], [[2, 2]]]}
    %0 = affine.load %arg0[%arg1 - 2] : memref<?xi32>
    // CHECK{LITERAL}: affine.load %arg0[%arg1 - 1] {dependences = [[[1, 1]]]}
    %1 = affine.load %arg0[%arg1 - 1] : memref<?xi32>
    %2 = addi %0, %1 : i32
    // CHECK: affine.store %2, %arg0[%arg1] {dependences = []}
    affine.store %2, %arg0[%arg1] : memref<?xi32>
  }
  return
}
