// RUN: circt-opt %s -memory-banking="factor=3 dimension=1" | FileCheck %s --check-prefix ATTRIBUTE

// ATTRIBUTE: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d1 mod 5)>
// ATTRIBUTE: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d1 floordiv 5)>
// ATTRIBUTE: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0 mod 5)>
// ATTRIBUTE: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0 floordiv 5)>

// ATTRIBUTE-LABEL:   func.func @main(
// ATTRIBUTE-SAME:                    %[[VAL_0:arg0]]: memref<3x1xf32>,
// ATTRIBUTE-SAME:                    %[[VAL_1:arg1]]: memref<3x1xf32>,
// ATTRIBUTE-SAME:                    %[[VAL_2:arg2]]: memref<3x1xf32>,
// ATTRIBUTE-SAME:                    %[[VAL_3:arg3]]: memref<3x1xf32>,
// ATTRIBUTE-SAME:                    %[[VAL_4:arg4]]: memref<3x1xf32>,
// ATTRIBUTE-SAME:                    %[[VAL_5:arg5]]: memref<1x3xf32>,
// ATTRIBUTE-SAME:                    %[[VAL_6:arg6]]: memref<1x3xf32>,
// ATTRIBUTE-SAME:                    %[[VAL_7:arg7]]: memref<1x3xf32>,
// ATTRIBUTE-SAME:                    %[[VAL_8:arg8]]: memref<1x3xf32>,
// ATTRIBUTE-SAME:                    %[[VAL_9:arg9]]: memref<1x3xf32>) -> (memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>) {
// ATTRIBUTE:           %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f32
// ATTRIBUTE:           %[[VAL_11:.*]] = memref.alloc() : memref<1x3xf32>
// ATTRIBUTE:           %[[VAL_12:.*]] = memref.alloc() : memref<1x3xf32>
// ATTRIBUTE:           %[[VAL_13:.*]] = memref.alloc() : memref<1x3xf32>
// ATTRIBUTE:           %[[VAL_14:.*]] = memref.alloc() : memref<1x3xf32>
// ATTRIBUTE:           %[[VAL_15:.*]] = memref.alloc() : memref<1x3xf32>
// ATTRIBUTE:           affine.parallel (%[[VAL_16:.*]]) = (0) to (5) {
// ATTRIBUTE:             affine.parallel (%[[VAL_17:.*]]) = (0) to (3) {
// ATTRIBUTE:               %[[VAL_18:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_17]], %[[VAL_16]])
// ATTRIBUTE:               %[[VAL_19:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_17]], %[[VAL_16]])
// ATTRIBUTE:               %[[VAL_20:.*]] = scf.index_switch %[[VAL_18]] -> f32
// ATTRIBUTE:               case 0 {
// ATTRIBUTE:                 %[[VAL_21:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_21]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               case 1 {
// ATTRIBUTE:                 %[[VAL_22:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_22]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               case 2 {
// ATTRIBUTE:                 %[[VAL_23:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_23]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               case 3 {
// ATTRIBUTE:                 %[[VAL_24:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_24]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               case 4 {
// ATTRIBUTE:                 %[[VAL_25:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_25]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               default {
// ATTRIBUTE:                 scf.yield %[[VAL_10]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               %[[VAL_26:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_16]], %[[VAL_17]])
// ATTRIBUTE:               %[[VAL_27:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_16]], %[[VAL_17]])
// ATTRIBUTE:               %[[VAL_28:.*]] = scf.index_switch %[[VAL_26]] -> f32
// ATTRIBUTE:               case 0 {
// ATTRIBUTE:                 %[[VAL_29:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_29]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               case 1 {
// ATTRIBUTE:                 %[[VAL_30:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_30]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               case 2 {
// ATTRIBUTE:                 %[[VAL_31:.*]] = affine.load %[[VAL_7]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_31]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               case 3 {
// ATTRIBUTE:                 %[[VAL_32:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_32]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               case 4 {
// ATTRIBUTE:                 %[[VAL_33:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield %[[VAL_33]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               default {
// ATTRIBUTE:                 scf.yield %[[VAL_10]] : f32
// ATTRIBUTE:               }
// ATTRIBUTE:               %[[VAL_34:.*]] = arith.mulf %[[VAL_20]], %[[VAL_28]] : f32
// ATTRIBUTE:               %[[VAL_35:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_16]], %[[VAL_17]])
// ATTRIBUTE:               %[[VAL_36:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_16]], %[[VAL_17]])
// ATTRIBUTE:               scf.index_switch %[[VAL_35]]
// ATTRIBUTE:               case 0 {
// ATTRIBUTE:                 affine.store %[[VAL_34]], %[[VAL_11]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield
// ATTRIBUTE:               }
// ATTRIBUTE:               case 1 {
// ATTRIBUTE:                 affine.store %[[VAL_34]], %[[VAL_12]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield
// ATTRIBUTE:               }
// ATTRIBUTE:               case 2 {
// ATTRIBUTE:                 affine.store %[[VAL_34]], %[[VAL_13]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield
// ATTRIBUTE:               }
// ATTRIBUTE:               case 3 {
// ATTRIBUTE:                 affine.store %[[VAL_34]], %[[VAL_14]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield
// ATTRIBUTE:               }
// ATTRIBUTE:               case 4 {
// ATTRIBUTE:                 affine.store %[[VAL_34]], %[[VAL_15]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// ATTRIBUTE:                 scf.yield
// ATTRIBUTE:               }
// ATTRIBUTE:               default {
// ATTRIBUTE:               }
// ATTRIBUTE:             }
// ATTRIBUTE:           }
// ATTRIBUTE:           return %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]] : memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>
// ATTRIBUTE:         }

module {
  func.func @main(%arg0: memref<3x5xf32> {banking.factor=5, banking.dimension=1}, %arg1: memref<5x3xf32>{banking.factor=5, banking.dimension=0}) -> (memref<5x3xf32>) {
    %mem = memref.alloc() {banking.factor=5, banking.dimension=0} : memref<5x3xf32>
    affine.parallel (%i) = (0) to (5) {
      affine.parallel (%j) = (0) to (3) {
        %1 = affine.load %arg0[%j, %i] : memref<3x5xf32>
        %2 = affine.load %arg1[%i, %j] : memref<5x3xf32>
        %3 = arith.mulf %1, %2 : f32
        affine.store %3, %mem[%i, %j] : memref<5x3xf32>
      }
    }
    return %mem : memref<5x3xf32>
  }
}
