// RUN: circt-opt --calyx-affine-to-scf --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:                    %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x6xf32>,
// CHECK-SAME:                    %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x6xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           scf.parallel (%[[VAL_9:.*]], %[[VAL_10:.*]]) = (%[[VAL_7]], %[[VAL_8]]) to (%[[VAL_5]], %[[VAL_6]]) step (%[[VAL_3]], %[[VAL_4]]) {
// CHECK:             scf.execute_region {
// CHECK:               memref.store %[[VAL_0]], %[[VAL_1]]{{\[}}%[[VAL_9]], %[[VAL_10]]] : memref<4x6xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               memref.store %[[VAL_0]], %[[VAL_2]]{{\[}}%[[VAL_10]], %[[VAL_9]]] : memref<4x6xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.reduce
// CHECK:           } {calyx.unroll = true}
// CHECK:           return
// CHECK:         }

module {
  func.func @main(%arg0: f32, %arg1: memref<4x6xf32>, %arg2: memref<4x6xf32>) {
    affine.parallel (%arg3, %arg4) = (0, 0) to (2, 3) {
      scf.execute_region {
        memref.store %arg0, %arg1[%arg3, %arg4] : memref<4x6xf32>
        scf.yield
      }
      scf.execute_region {
        memref.store %arg0, %arg2[%arg4, %arg3] : memref<4x6xf32>
        scf.yield
      }
    } {calyx.unroll = true}
    return
  }
}
