// RUN: circt-opt -split-input-file --flatten-memref-calls %s | FileCheck %s


// CHECK-LABEL:   func private @foo(memref<900xi32>) -> i32

// CHECK-LABEL:   func @main() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<1x30x30xi32>
// CHECK:           %[[VAL_2:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_0]], 0, 0] [1, 30, 30] [1, 1, 1] : memref<1x30x30xi32> to memref<30x30xi32>
// CHECK:           %[[VAL_3:.*]] = memref.subview %[[VAL_2]][0, 0] [1, 900] [1, 1] : memref<30x30xi32> to memref<900xi32>
// CHECK:           %[[VAL_4:.*]] = call @foo(%[[VAL_3]]) : (memref<900xi32>) -> i32
// CHECK:           return
// CHECK:         }
module  {
  func private @foo(memref<30x30xi32>) -> i32
  func @main() {
    %c0 = arith.constant 0 : index
    %3 = memref.alloca() : memref<1x30x30xi32>
    %4 = memref.subview %3[%c0, 0, 0] [1, 30, 30] [1, 1, 1] : memref<1x30x30xi32> to memref<30x30xi32>
    %7 = call @foo(%4) : (memref<30x30xi32>) -> i32
    return
  }
}
