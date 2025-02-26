// RUN: circt-opt %s -memory-banking="factors=2 dimensions=1" --canonicalize --split-input-file | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 mod 2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0 mod 3)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0 mod 4)>

// CHECK-LABEL:   func.func @multi_config(
// CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<8x3xf32>,
// CHECK-SAME:                            %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<8x3xf32>) -> (memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>) {
// CHECK:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_12:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_13:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_14:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_15:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_16:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           affine.parallel (%[[VAL_17:.*]]) = (0) to (8) {
// CHECK:             affine.parallel (%[[VAL_18:.*]]) = (0) to (6) {
// CHECK:               %[[VAL_19:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_17]])
// CHECK:               %[[VAL_20:.*]] = scf.index_switch %[[VAL_19]] -> f32
// CHECK:               case 0 {
// CHECK:                 %[[VAL_21:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_18]])
// CHECK:                 %[[VAL_22:.*]] = scf.index_switch %[[VAL_21]] -> f32
// CHECK:                 case 0 {
// CHECK:                   %[[VAL_23:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_17]] floordiv 2, %[[VAL_18]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_23]] : f32
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   %[[VAL_24:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_17]] floordiv 2, %[[VAL_18]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_24]] : f32
// CHECK:                 }
// CHECK:                 case 2 {
// CHECK:                   %[[VAL_25:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_17]] floordiv 2, %[[VAL_18]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_25]] : f32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_8]] : f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_22]] : f32
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 %[[VAL_26:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_18]])
// CHECK:                 %[[VAL_27:.*]] = scf.index_switch %[[VAL_26]] -> f32
// CHECK:                 case 0 {
// CHECK:                   %[[VAL_28:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_17]] floordiv 2, %[[VAL_18]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_28]] : f32
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   %[[VAL_29:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_17]] floordiv 2, %[[VAL_18]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_29]] : f32
// CHECK:                 }
// CHECK:                 case 2 {
// CHECK:                   %[[VAL_30:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_17]] floordiv 2, %[[VAL_18]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_30]] : f32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_8]] : f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_27]] : f32
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_8]] : f32
// CHECK:               }
// CHECK:               %[[VAL_31:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_18]])
// CHECK:               %[[VAL_32:.*]] = scf.index_switch %[[VAL_31]] -> f32
// CHECK:               case 0 {
// CHECK:                 %[[VAL_33:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_17]], %[[VAL_18]] floordiv 2] : memref<8x3xf32>
// CHECK:                 scf.yield %[[VAL_33]] : f32
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 %[[VAL_34:.*]] = affine.load %[[VAL_7]]{{\[}}%[[VAL_17]], %[[VAL_18]] floordiv 2] : memref<8x3xf32>
// CHECK:                 scf.yield %[[VAL_34]] : f32
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_8]] : f32
// CHECK:               }
// CHECK:               %[[VAL_35:.*]] = arith.mulf %[[VAL_20]], %[[VAL_32]] : f32
// CHECK:               %[[VAL_36:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_17]])
// CHECK:               scf.index_switch %[[VAL_36]]
// CHECK:               case 0 {
// CHECK:                 %[[VAL_37:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_18]])
// CHECK:                 scf.index_switch %[[VAL_37]]
// CHECK:                 case 0 {
// CHECK:                   affine.store %[[VAL_35]], %[[VAL_9]]{{\[}}%[[VAL_17]] floordiv 4, %[[VAL_18]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   affine.store %[[VAL_35]], %[[VAL_10]]{{\[}}%[[VAL_17]] floordiv 4, %[[VAL_18]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 default {
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 %[[VAL_38:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_18]])
// CHECK:                 scf.index_switch %[[VAL_38]]
// CHECK:                 case 0 {
// CHECK:                   affine.store %[[VAL_35]], %[[VAL_11]]{{\[}}%[[VAL_17]] floordiv 4, %[[VAL_18]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   affine.store %[[VAL_35]], %[[VAL_12]]{{\[}}%[[VAL_17]] floordiv 4, %[[VAL_18]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 default {
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 %[[VAL_39:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_18]])
// CHECK:                 scf.index_switch %[[VAL_39]]
// CHECK:                 case 0 {
// CHECK:                   affine.store %[[VAL_35]], %[[VAL_13]]{{\[}}%[[VAL_17]] floordiv 4, %[[VAL_18]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   affine.store %[[VAL_35]], %[[VAL_14]]{{\[}}%[[VAL_17]] floordiv 4, %[[VAL_18]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 default {
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 3 {
// CHECK:                 %[[VAL_40:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_18]])
// CHECK:                 scf.index_switch %[[VAL_40]]
// CHECK:                 case 0 {
// CHECK:                   affine.store %[[VAL_35]], %[[VAL_15]]{{\[}}%[[VAL_17]] floordiv 4, %[[VAL_18]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   affine.store %[[VAL_35]], %[[VAL_16]]{{\[}}%[[VAL_17]] floordiv 4, %[[VAL_18]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 default {
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               default {
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]] : memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>
// CHECK:         }

module {
  func.func @multi_config(%arg0: memref<8x6xf32> {banking.factors=[2, 3], banking.dimensions=[0, 1]}, %arg1: memref<8x6xf32>) -> memref<8x6xf32> {
    %mem = memref.alloc() {banking.factors=[4, 2], banking.dimensions=[0, 1]} : memref<8x6xf32>
    affine.parallel (%i) = (0) to (8) {
      affine.parallel (%j) = (0) to (6) {
        %1 = affine.load %arg0[%i, %j] : memref<8x6xf32>
        %2 = affine.load %arg1[%i, %j] : memref<8x6xf32>
        %3 = arith.mulf %1, %2 : f32
        affine.store %3, %mem[%i, %j] : memref<8x6xf32>
      }
    }
    return %mem : memref<8x6xf32>
  }
}
