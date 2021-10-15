// RUN: circt-opt %s -test-scheduling-analysis | FileCheck %s

// CHECK-LABEL: func @test1
func @test1(%arg0: memref<?xi32>) -> i32 {
  %c0_i32 = constant 0 : i32
  %0:2 = affine.for %arg1 = 0 to 10 iter_args(%arg2 = %c0_i32, %arg3 = %c0_i32) -> (i32, i32) {
    %1 = affine.load %arg0[%arg1] : memref<?xi32>
    // CHECK: addi %arg2, %1 {dependence}
    %2 = addi %arg2, %1 : i32
    affine.yield %2, %2 : i32, i32
  }
  return %0#1 : i32
}
// CHECK-LABEL: func @test2
#set = affine_set<(d0) : (d0 - 3 >= 0)>
func @test2(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
  affine.for %arg2 = 0 to 10 {
    %0 = affine.load %arg0[%arg2] : memref<?xi32>
    affine.if #set(%arg2) {
      // CHECK: affine.load %arg0[%arg2 - 3] {dependence}
      %1 = affine.load %arg0[%arg2 - 3] : memref<?xi32>
      %2 = addi %0, %1 : i32
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
    // CHECK: affine.load %2[0] {dependence}
    %3 = affine.load %2[0] : memref<1xi32>
    // CHECK: affine.load %1[0] {dependence}
    %4 = affine.load %1[0] : memref<1xi32>
    // CHECK: affine.store %4, %2[0] {dependence}
    affine.store %4, %2[0] : memref<1xi32>
    // CHECK: affine.load %0[0] {dependence}
    %5 = affine.load %0[0] : memref<1xi32>
    // CHECK: affine.store %5, %1[0] {dependence}
    affine.store %5, %1[0] : memref<1xi32>
    %6 = affine.load %arg0[%arg1] : memref<?xi32>
    %7 = addi %3, %6 : i32
    // CHECK: affine.store %7, %0[0] {dependence}
    affine.store %7, %0[0] : memref<1xi32>
  }
  return
}

// CHECK-LABEL: func @test4
// CHECK-NOT: dependence
func @test4(%arg0: memref<?xi32>, %arg1: memref<?xi32>) {
  %c1_i32 = constant 1 : i32
  affine.for %arg2 = 0 to 10 {
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
    // CHECK: affine.load %arg0[%arg1 - 2] {dependence}
    %0 = affine.load %arg0[%arg1 - 2] : memref<?xi32>
    // CHECK: affine.load %arg0[%arg1 - 1] {dependence}
    %1 = affine.load %arg0[%arg1 - 1] : memref<?xi32>
    %2 = addi %0, %1 : i32
    affine.store %2, %arg0[%arg1] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func @test6
#set1 = affine_set<(d0) : (d0 - 5 >= 0)>
func @test6(%arg0: memref<?xi32>) {
  affine.for %arg1 = 0 to 10 {
    %0 = affine.if #set1(%arg1) -> i32 {
      %1 = affine.load %arg0[%arg1] : memref<?xi32>
      affine.yield %1 : i32
    } else {
      %c1_i32 = constant 1 : i32
      affine.yield %c1_i32 : i32
    }
    // CHECK: } {dependence}
    // CHECK: affine.store %0, %arg0[%arg1] {dependence}
    affine.store %0, %arg0[%arg1] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func @test7
#set2 = affine_set<(d0) : (d0 - 2 >= 0)>
#set3 = affine_set<(d0) : (d0 - 6 >= 0)>
func @test7(%arg0: memref<?xi32>) {
  affine.for %arg1 = 0 to 10 {
    affine.if #set2(%arg1) {
      %0 = affine.if #set3(%arg1) -> i32 {
        %1 = affine.load %arg0[%arg1] : memref<?xi32>
        affine.yield %1 : i32
      }
      // CHECK: } {dependence}
      // CHECK: affine.store %0, %arg0[%arg1] {dependence}
      affine.store %0, %arg0[%arg1] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func @test8
func @test8(%arg0: memref<?xi32>) {
  affine.for %arg1 = 0 to 10 {
    affine.if #set2(%arg1) {
      %0 = affine.if #set3(%arg1) -> i32 {
        %1 = affine.load %arg0[%arg1] : memref<?xi32>
        affine.yield %1 : i32
      } else {
        %1 = affine.load %arg0[%arg1] : memref<?xi32>
        affine.yield %1 : i32
      }
      // CHECK: } {dependence}
      // CHECK: affine.store %0, %arg0[%arg1] {dependence}
      affine.store %0, %arg0[%arg1] : memref<?xi32>
    }
  }
  return
}

// CHECK-LABEL: func @test9
func @test9(%arg0: memref<4x4xi32>, %arg1: memref<4x4xi32>, %arg2: memref<4x4xi32>) {
  affine.for %arg3 = 0 to 4 {
    affine.for %arg4 = 0 to 4 {
      affine.for %arg5 = 0 to 4 {
        %0 = affine.load %arg0[%arg3, %arg5] : memref<4x4xi32>
        %1 = affine.load %arg1[%arg5, %arg4] : memref<4x4xi32>
        %2 = affine.load %arg2[%arg3, %arg4] : memref<4x4xi32>
        %3 = muli %0, %1 : i32
        %4 = addi %2, %3 : i32
        affine.store %4, %arg2[%arg3, %arg4] : memref<4x4xi32>
      }
    }
  }
  return
}
