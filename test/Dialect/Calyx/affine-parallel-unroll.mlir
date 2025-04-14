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

// -----

// CHECK-LABEL:   func.func @licm(
// CHECK-SAME:                    %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2xf32>,
// CHECK-SAME:                    %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2xf32>,
// CHECK-SAME:                    %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2xf32>) {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<2xf32>
// CHECK-DAG:           %[[VAL_4:.*]] = affine.load %[[VAL_0]][0] : memref<2xf32>
// CHECK-DAG:           %[[VAL_5:.*]] = affine.load %[[VAL_1]][0] : memref<2xf32>
// CHECK:           affine.parallel (%[[VAL_6:.*]]) = (0) to (1) {
// CHECK:             scf.execute_region {
// CHECK:               affine.for %[[VAL_7:.*]] = 0 to 2 {
// CHECK:                 %[[VAL_8:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_7]]] : memref<2xf32>
// CHECK:                 %[[VAL_9:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:                 %[[VAL_10:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f32
// CHECK:                 affine.store %[[VAL_10]], %[[VAL_2]]{{\[}}%[[VAL_7]]] : memref<2xf32>
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               affine.for %[[VAL_11:.*]] = 0 to 2 {
// CHECK:                 %[[VAL_12:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_11]]] : memref<2xf32>
// CHECK:                 %[[VAL_13:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:                 %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK:                 affine.store %[[VAL_14]], %[[VAL_3]]{{\[}}%[[VAL_11]]] : memref<2xf32>
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           } {calyx.unroll = true}
// CHECK:           return
// CHECK:         }

#map = affine_map<(d0) -> (d0 floordiv 2)>
module {
  func.func @licm(%arg0: memref<2xf32>, %arg1: memref<2xf32>, %arg2: memref<2xf32>) {
    %alloc = memref.alloc() : memref<2xf32>
    %cst = arith.constant 0.0 : f32
    affine.parallel (%iv_par) = (0) to (2) {
      %0 = affine.apply #map(%iv_par)
      affine.for %iv_for = 0 to 2 {
        // The following load will be hoisted.
        %1 = affine.load %arg0[%0] : memref<2xf32>
        // The following load will be hoisted.
        %2 = affine.load %arg1[%0] : memref<2xf32>
        %3 = scf.index_switch %iv_par -> f32
        case 0 {
          %loaded = affine.load %alloc[%iv_for] : memref<2xf32>
          scf.yield %loaded : f32
        }
        case 1 {
          %loaded = affine.load %arg2[%iv_for] : memref<2xf32>
          scf.yield %loaded : f32
        }
        default {
          scf.yield %cst : f32
        }
        %4 = arith.mulf %1, %2 : f32
        %5 = arith.addf %3, %4 : f32
        scf.index_switch %iv_par
        case 0 {
          affine.store %5, %alloc[%iv_for] : memref<2xf32>
          scf.yield
        }
        case 1 {
          affine.store %5, %arg2[%iv_for] : memref<2xf32>
          scf.yield
        }
        default {
          scf.yield
        }
      }
    }
    return
  }
}

// -----

// We do not hoist a constant-indices load when there is a write in the parallel region

// CHECK-LABEL:   func.func @war(
// CHECK-SAME:                   %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2xf32>,
// CHECK-SAME:                   %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2xf32>,
// CHECK-SAME:                   %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2xf32>,
// CHECK-SAME:                   %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2xf32>) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 4.200000e+01 : f32
// CHECK:           affine.parallel (%[[VAL_5:.*]]) = (0) to (1) {
// CHECK:             scf.execute_region {
// CHECK:               affine.for %[[VAL_6:.*]] = 0 to 2 {
// CHECK:                 affine.store %[[VAL_4]], %[[VAL_1]]{{\[}}%[[VAL_6]]] : memref<2xf32>
// CHECK:                 %[[VAL_7:.*]] = affine.load %[[VAL_0]][0] : memref<2xf32>
// CHECK:                 affine.store %[[VAL_7]], %[[VAL_3]]{{\[}}%[[VAL_6]]] : memref<2xf32>
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               affine.for %[[VAL_8:.*]] = 0 to 2 {
// CHECK:                 affine.store %[[VAL_4]], %[[VAL_0]]{{\[}}%[[VAL_8]]] : memref<2xf32>
// CHECK:                 %[[VAL_9:.*]] = affine.load %[[VAL_0]][0] : memref<2xf32>
// CHECK:                 affine.store %[[VAL_9]], %[[VAL_2]]{{\[}}%[[VAL_8]]] : memref<2xf32>
// CHECK:               }
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           } {calyx.unroll = true}
// CHECK:           return
// CHECK:         }

module {
  func.func @war(%arg0: memref<2xf32>, %arg1: memref<2xf32>, %arg2: memref<2xf32>, %arg3: memref<2xf32>) {
    %cst = arith.constant 4.200000e+01 : f32
    affine.parallel (%arg4) = (0) to (1) {
      scf.execute_region {
        affine.for %arg5 = 0 to 2 {
          affine.store %cst, %arg1[%arg5] : memref<2xf32>
          %0 = affine.load %arg0[0] : memref<2xf32>
          affine.store %0, %arg3[%arg5] : memref<2xf32>
        }
        scf.yield
      }
      scf.execute_region {
        affine.for %arg5 = 0 to 2 {
          affine.store %cst, %arg0[%arg5] : memref<2xf32>
          %0 = affine.load %arg0[0] : memref<2xf32>
          affine.store %0, %arg2[%arg5] : memref<2xf32>
        }
        scf.yield
      }
    } {calyx.unroll = true}
    return
  }
}

// -----

// Test non-conflicting constant writes after canonicalizing `scf.index_switch`

// CHECK-LABEL:   func.func @const_writes(
// CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<1x1xf32>,
// CHECK-SAME:                            %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<1x1xf32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 4.200000e+01 : f32
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<1x1xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<1x1xf32>
// CHECK:           affine.parallel (%[[VAL_5:.*]]) = (0) to (1) {
// CHECK:             scf.execute_region {
// CHECK:               affine.store %[[VAL_2]], %[[VAL_4]][0, 0] : memref<1x1xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               affine.store %[[VAL_2]], %[[VAL_1]][0, 0] : memref<1x1xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               affine.store %[[VAL_2]], %[[VAL_3]][0, 0] : memref<1x1xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:             scf.execute_region {
// CHECK:               affine.store %[[VAL_2]], %[[VAL_0]][0, 0] : memref<1x1xf32>
// CHECK:               scf.yield
// CHECK:             }
// CHECK:           } {calyx.unroll = true}
// CHECK:           return
// CHECK:         }

#map = affine_map<(d0) -> (d0 mod 2)>

module {
  func.func @const_writes(%arg0: memref<1x1xf32>, %arg1: memref<1x1xf32>) {
    %fortytwo = arith.constant 42.0 : f32
    %zeroidx = arith.constant 0 : index
    %alloc_0 = memref.alloc() : memref<1x1xf32>
    %alloc_1 = memref.alloc() : memref<1x1xf32>
    affine.parallel (%arg2, %arg3) = (0, 0) to (2, 2) {
      %0 = affine.apply #map(%arg2)
      scf.index_switch %0 
        case 0 {
          %1 = affine.apply #map(%arg3)
          scf.index_switch %1 
          case 0 {
            affine.store %fortytwo, %arg0[%arg2 floordiv 2, %arg3 floordiv 2] : memref<1x1xf32>
            scf.yield
          }
          case 1 {
            affine.store %fortytwo, %alloc_0[%arg2 floordiv 2, %arg3 floordiv 2] : memref<1x1xf32>
            scf.yield
          }
          default {
          }
          scf.yield
        }
        case 1 {
          %1 = affine.apply #map(%arg3)
          scf.index_switch %1 
          case 0 {
            affine.store %fortytwo, %arg1[%arg2 floordiv 2, %arg3 floordiv 2] : memref<1x1xf32>
            scf.yield
          }
          case 1 {
            affine.store %fortytwo, %alloc_1[%arg2 floordiv 2, %arg3 floordiv 2] : memref<1x1xf32>
            scf.yield
          }
          default {
          }
          scf.yield
        }
        default {
        }
    }
    return
  }
}

// -----

// Test hosting read when the access indices depend on outer nested for-loops.

// CHECK-LABEL:   func.func @hoist_read(
// CHECK-SAME:                          %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x2xf32>,
// CHECK-SAME:                          %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x2xf32>,
// CHECK-SAME:                          %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x2xf32>,
// CHECK-SAME:                          %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x2xf32>,
// CHECK-SAME:                          %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x2xf32>,
// CHECK-SAME:                          %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x2xf32>,
// CHECK-SAME:                          %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x2xf32>,
// CHECK-SAME:                          %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x2xf32>,
// CHECK-SAME:                          %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32) {
// CHECK:           affine.for %[[VAL_9:.*]] = 0 to 4 {
// CHECK:             affine.for %[[VAL_10:.*]] = 0 to 2 {
// CHECK-DAG:               %[[VAL_11:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_10]], %[[VAL_9]] floordiv 2] : memref<2x2xf32>
// CHECK-DAG:               %[[VAL_12:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_10]], %[[VAL_9]] floordiv 2] : memref<2x2xf32>
// CHECK:               affine.parallel (%[[VAL_13:.*]]) = (0) to (1) {
// CHECK:                 scf.execute_region {
// CHECK:                   %[[VAL_14:.*]] = arith.mulf %[[VAL_11]], %[[VAL_8]] : f32
// CHECK:                   affine.store %[[VAL_14]], %[[VAL_3]]{{\[}}%[[VAL_10]], %[[VAL_9]] floordiv 2] : memref<2x2xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 scf.execute_region {
// CHECK:                   %[[VAL_15:.*]] = arith.mulf %[[VAL_12]], %[[VAL_8]] : f32
// CHECK:                   affine.store %[[VAL_15]], %[[VAL_2]]{{\[}}%[[VAL_10]], %[[VAL_9]] floordiv 2] : memref<2x2xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 scf.execute_region {
// CHECK:                   %[[VAL_16:.*]] = arith.mulf %[[VAL_12]], %[[VAL_8]] : f32
// CHECK:                   affine.store %[[VAL_16]], %[[VAL_1]]{{\[}}%[[VAL_10]], %[[VAL_9]] floordiv 2] : memref<2x2xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 scf.execute_region {
// CHECK:                   %[[VAL_17:.*]] = arith.mulf %[[VAL_11]], %[[VAL_8]] : f32
// CHECK:                   affine.store %[[VAL_17]], %[[VAL_0]]{{\[}}%[[VAL_10]], %[[VAL_9]] floordiv 2] : memref<2x2xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:               } {calyx.unroll = true}
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

module {
  func.func @hoist_read(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>, %arg3: memref<2x2xf32>, %arg4: memref<2x2xf32>, %arg5: memref<2x2xf32>, %arg6: memref<2x2xf32>, %arg7: memref<2x2xf32>, %arg8: f32) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg9 = 0 to 4 {
      affine.for %arg10 = 0 to 2 {
        affine.parallel (%arg11, %arg12) = (0, 0) to (2, 2) {
          %0 = scf.index_switch %arg11 -> f32
          case 0 {
            %2 = scf.index_switch %arg12 -> f32
            case 0 {
              // This load will be hoisted after unrolling and simplifying the access indices.
              %3 = affine.load %arg4[%arg10 + %arg11 floordiv 2, %arg9 floordiv 2] : memref<2x2xf32>
              scf.yield %3 : f32
            }
            case 1 {
              %3 = affine.load %arg5[%arg10 + %arg11 floordiv 2, %arg9 floordiv 2] : memref<2x2xf32>
              scf.yield %3 : f32
            }
            default {
              scf.yield %cst : f32
            }
            scf.yield %2 : f32
          }
          case 1 {
            %2 = scf.index_switch %arg12 -> f32
            case 0 {
              // This load will be hoisted after unrolling and simplifying the access indices.
              %3 = affine.load %arg5[%arg10 + %arg11 floordiv 2, %arg9 floordiv 2] : memref<2x2xf32>
              scf.yield %3 : f32
            }
            case 1 {
              %3 = affine.load %arg4[%arg10 + %arg11 floordiv 2, %arg9 floordiv 2] : memref<2x2xf32>
              scf.yield %3 : f32
            }
            default {
              scf.yield %cst : f32
            }
            scf.yield %2 : f32
          }
          default {
            scf.yield %cst : f32
          }
          %1 = arith.mulf %0, %arg8 : f32
          scf.index_switch %arg11
          case 0 {
            scf.index_switch %arg12
            case 0 {
              affine.store %1, %arg0[%arg10 + %arg11 floordiv 2, %arg9 floordiv 2] : memref<2x2xf32>
              scf.yield
            }
            case 1 {
              affine.store %1, %arg1[%arg10 + %arg11 floordiv 2, %arg9 floordiv 2] : memref<2x2xf32>
              scf.yield
            }
            default {
            }
            scf.yield
          }
          case 1 {
            scf.index_switch %arg12
            case 0 {
              affine.store %1, %arg2[%arg10 + %arg11 floordiv 2, %arg9 floordiv 2] : memref<2x2xf32>
              scf.yield
            }
            case 1 {
              affine.store %1, %arg3[%arg10 + %arg11 floordiv 2, %arg9 floordiv 2] : memref<2x2xf32>
              scf.yield
            }
            default {
            }
            scf.yield
          }
          default {
          }
        }
      }
    }
    return
  }
}

