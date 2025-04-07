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
// CHECK:           } {unparallelized}
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
// CHECK:               } {unparallelized}
// CHECK:             }
// CHECK:           } {unparallelized}
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

// -----

// Test simplify `scf.index_switch`

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<6xf32>,
// CHECK-SAME:                    %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<6xf32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<6xf32>
// CHECK:           affine.for %[[VAL_4:.*]] = 0 to 18 step 3 {
// CHECK:             affine.parallel (%[[VAL_5:.*]]) = (0) to (3) {
// CHECK:               scf.index_switch %[[VAL_5]]
// CHECK:               case 0 {
// CHECK:                 affine.store %[[VAL_2]], %[[VAL_0]][0] : memref<6xf32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 affine.store %[[VAL_2]], %[[VAL_1]][0] : memref<6xf32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 affine.store %[[VAL_2]], %[[VAL_3]][0] : memref<6xf32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               default {
// CHECK:               }
// CHECK:             }
// CHECK:           } {unparallelized}
// CHECK:           return
// CHECK:         }

#map = affine_map<(d0) -> (d0 mod 3)>
module {
  func.func @main(%arg0: memref<6xf32>, %arg1: memref<6xf32>) {
    %c0 = arith.constant 0.0 : f32
    %alloc = memref.alloc() : memref<6xf32>
    affine.parallel (%arg2) = (0) to (18) {
      %1 = affine.apply #map(%arg2)
      scf.index_switch %1
      case 0 {
        affine.store %c0, %arg0[%1 floordiv 3] : memref<6xf32>
        scf.yield
      }
      case 1 {
        affine.store %c0, %arg1[%1 floordiv 3] : memref<6xf32>
        scf.yield
      }
      case 2 {
        affine.store %c0, %alloc[%1 floordiv 3] : memref<6xf32>
        scf.yield
      }
      default {
      }
    } {unparallelize.factor=3}
    return
  }
}
