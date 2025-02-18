// RUN: circt-opt --affine-parallel-unroll --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:.*]]: memref<2x2xf32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: memref<2x2xf32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<2x2xf32>
// CHECK:           affine.parallel (%[[VAL_5:.*]]) = (0) to (1) {
// CHECK:             scf.execute_region {
// CHECK:               %[[VAL_6:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]]] : memref<2x2xf32>
// CHECK:               %[[VAL_7:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]], %[[VAL_3]]] : memref<2x2xf32>
// CHECK:               %[[VAL_8:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f32
// CHECK:               memref.store %[[VAL_8]], %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]] : memref<2x2xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               %[[VAL_9:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_2]]] : memref<2x2xf32>
// CHECK:               %[[VAL_10:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]], %[[VAL_2]]] : memref<2x2xf32>
// CHECK:               %[[VAL_11:.*]] = arith.addf %[[VAL_9]], %[[VAL_10]] : f32
// CHECK:               memref.store %[[VAL_11]], %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_2]]] : memref<2x2xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               %[[VAL_12:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_2]], %[[VAL_3]]] : memref<2x2xf32>
// CHECK:               %[[VAL_13:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]], %[[VAL_3]]] : memref<2x2xf32>
// CHECK:               %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:               memref.store %[[VAL_14]], %[[VAL_1]]{{\[}}%[[VAL_2]], %[[VAL_3]]] : memref<2x2xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               %[[VAL_15:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_2]], %[[VAL_2]]] : memref<2x2xf32>
// CHECK:               %[[VAL_16:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]], %[[VAL_2]]] : memref<2x2xf32>
// CHECK:               %[[VAL_17:.*]] = arith.addf %[[VAL_15]], %[[VAL_16]] : f32
// CHECK:               memref.store %[[VAL_17]], %[[VAL_1]]{{\[}}%[[VAL_2]], %[[VAL_2]]] : memref<2x2xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           } {calyx.unroll = true}
// CHECK:           return
// CHECK:         }

module {
  func.func @main(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
    %alloc = memref.alloc() : memref<2x2xf32>
    affine.parallel (%i, %j) = (0, 0) to (2, 2) {
      %0 = memref.load %arg0[%i, %j] : memref<2x2xf32>
      %1 = memref.load %alloc[%i, %j] : memref<2x2xf32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %arg1[%i, %j] : memref<2x2xf32>
    }
    return
  }
}

// -----

// Test parallel op lowering when it has region-based nested ops, such as `affine.for`

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:.*]]: memref<2x2xf32>,
// CHECK-SAME:                    %[[VAL_1:.*]]: memref<2x2xf32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<2x2xf32>
// CHECK:           affine.parallel (%[[VAL_5:.*]]) = (0) to (1) {
// CHECK:             scf.execute_region {
// CHECK:               affine.for %[[VAL_6:.*]] = 0 to 2 {
// CHECK:                 %[[VAL_7:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_6]]] : memref<2x2xf32>
// CHECK:                 %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]], %[[VAL_6]]] : memref<2x2xf32>
// CHECK:                 %[[VAL_9:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:                 memref.store %[[VAL_9]], %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_6]]] : memref<2x2xf32>
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               affine.for %[[VAL_10:.*]] = 0 to 2 {
// CHECK:                 %[[VAL_11:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_2]], %[[VAL_10]]] : memref<2x2xf32>
// CHECK:                 %[[VAL_12:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]], %[[VAL_10]]] : memref<2x2xf32>
// CHECK:                 %[[VAL_13:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] : f32
// CHECK:                 memref.store %[[VAL_13]], %[[VAL_1]]{{\[}}%[[VAL_2]], %[[VAL_10]]] : memref<2x2xf32>
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           } {calyx.unroll = true}
// CHECK:           return
// CHECK:         }

module {
  func.func @main(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
    %alloc = memref.alloc() : memref<2x2xf32>
    affine.parallel (%i) = (0) to (2) {
      affine.for %j = 0 to 2 {
        %0 = memref.load %arg0[%i, %j] : memref<2x2xf32>
        %1 = memref.load %alloc[%i, %j] : memref<2x2xf32>
        %2 = arith.addf %0, %1 : f32
        memref.store %2, %arg1[%i, %j] : memref<2x2xf32>
      }
    }
    return
  }
}

// -----

// Canonicalze SCF IndexSwitchOp after unroll as Affine ParallelOp often contains SCF IndexSwitchOp after banking

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:.*]]: memref<2x2xf32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 3.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 4.200000e+00 : f32
// CHECK:           affine.parallel (%[[VAL_7:.*]]) = (0) to (1) {
// CHECK:             scf.execute_region {
// CHECK:               memref.store %[[VAL_6]], %[[VAL_0]]{{\[}}%[[VAL_2]], %[[VAL_2]]] : memref<2x2xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               memref.store %[[VAL_4]], %[[VAL_0]]{{\[}}%[[VAL_2]], %[[VAL_1]]] : memref<2x2xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               memref.store %[[VAL_5]], %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_2]]] : memref<2x2xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               memref.store %[[VAL_3]], %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_1]]] : memref<2x2xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           } {calyx.unroll = true}
// CHECK:           return
// CHECK:         }

#map = affine_map<(d0) -> (d0 mod 2)>
module {
  func.func @main(%arg0: memref<2x2xf32>) {
    %cst_0 = arith.constant 0.0 : f32
    %cst_1 = arith.constant 1.0 : f32
    %cst_2 = arith.constant 2.0 : f32
    %cst_3 = arith.constant 3.0 : f32
    %cst_4 = arith.constant 4.2 : f32
    affine.parallel (%i, %j) = (0, 0) to (2, 2) {
      %0 = affine.apply #map(%j)
      %1 = scf.index_switch %0 -> f32
      case 0 {
        %3 = affine.apply #map(%i)
        %4 = scf.index_switch %3 -> f32
        case 0 {
          scf.yield %cst_1 : f32
        }
        case 1 {
          scf.yield %cst_2 : f32
        }
        default {
          scf.yield %cst_0 : f32
        }
        scf.yield %4 : f32
      }
      case 1 {
        %3 = affine.apply #map(%i)
        %4 = scf.index_switch %3 -> f32
        case 0 {
          scf.yield %cst_3 : f32
        }
        case 1 {
          scf.yield %cst_4 : f32
        }
        default {
          scf.yield %cst_0 : f32
        }
        scf.yield %4 : f32
      }
      default {
        scf.yield %cst_0 : f32
      }
      memref.store %1, %arg0[%i, %j] : memref<2x2xf32>
    }
    return
  }
}
