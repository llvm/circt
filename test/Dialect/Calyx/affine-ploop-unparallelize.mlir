// RUN: circt-opt --affine-ploop-unparallelize --canonicalize --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<16x4xf32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           affine.for %[[VAL_2:.*]] = 0 to 4 step 2 {
// CHECK:             affine.parallel (%[[VAL_3:.*]]) = (0) to (2) {
// CHECK:               affine.for %[[VAL_4:.*]] = 0 to 16 {
// CHECK:                 affine.store %[[VAL_1]], %[[VAL_0]]{{\[}}%[[VAL_2]] + %[[VAL_3]], %[[VAL_4]]] : memref<16x4xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

module {
  func.func @main(%arg0: memref<16x4xf32>) {
    %c0 = arith.constant 0.0 : f32
    affine.parallel (%arg2) = (0) to (4) {
      affine.for %arg3 = 0 to 16 {
        affine.store %c0, %arg0[%arg2, %arg3] : memref<16x4xf32>
      }
    } {unparallelize.factor=2}
    return
  }
}

// -----

// Test nested `affine.parallel`s

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<16x4xf32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           affine.for %[[VAL_2:.*]] = 0 to 4 step 2 {
// CHECK:             affine.parallel (%[[VAL_3:.*]]) = (0) to (2) {
// CHECK:               affine.for %[[VAL_4:.*]] = 0 to 16 {
// CHECK:                 affine.parallel (%[[VAL_5:.*]]) = (0) to (1) {
// CHECK:                   affine.store %[[VAL_1]], %[[VAL_0]]{{\[}}%[[VAL_2]] + %[[VAL_3]], %[[VAL_4]] + %[[VAL_5]]] : memref<16x4xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

module {
  func.func @main(%arg0: memref<16x4xf32>) {
    %c0 = arith.constant 0.0 : f32
    affine.parallel (%arg2) = (0) to (4) {
      affine.parallel (%arg3) = (0) to (16) {
        affine.store %c0, %arg0[%arg2, %arg3] : memref<16x4xf32>
      } {unparallelize.factor=1}
    } {unparallelize.factor=2}
    return
  }
}
