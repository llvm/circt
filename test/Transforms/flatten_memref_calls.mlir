// RUN: circt-opt -split-input-file --flatten-memref-calls %s | FileCheck %s

// CHECK-LABEL:   func private @foo(memref<900xi32>) -> i32

// CHECK-LABEL:       func @main() {
// CHECK:               %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_1:.*]] = memref.alloca() : memref<30x30xi32>
// CHECK:               %[[VAL_2:.*]] = memref.collapse_shape %[[VAL_1]]
// CHECK-SAME{LITERAL}:   [[0, 1]] : memref<30x30xi32> into memref<900xi32>
// CHECK:               %[[VAL_3:.*]] = call @foo(%[[VAL_2]]) : (memref<900xi32>) -> i32
// CHECK:               return
// CHECK:             }
module  {
  func.func private @foo(memref<30x30xi32>) -> i32
  func.func @main() {
    %c0 = arith.constant 0 : index
    %0 = memref.alloca() : memref<30x30xi32>
    %1 = call @foo(%0) : (memref<30x30xi32>) -> i32
    return
  }
}
