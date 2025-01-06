// RUN: circt-opt %s -split-input-file -memory-banking="banking-factor=2 dimension=1" | FileCheck %s --check-prefix RANK2-BANKDIM1

// RANK2-BANKDIM1: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d1 mod 2)>
// RANK2-BANKDIM1: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d1 floordiv 2)>

// RANK2-BANKDIM1-LABEL:   func.func @rank_two_bank_dim1(
// RANK2-BANKDIM1-SAME:                                  %[[VAL_0:arg0]]: memref<8x3xf32>,
// RANK2-BANKDIM1-SAME:                                  %[[VAL_1:arg1]]: memref<8x3xf32>,
// RANK2-BANKDIM1-SAME:                                  %[[VAL_2:arg2]]: memref<8x3xf32>,
// RANK2-BANKDIM1-SAME:                                  %[[VAL_3:arg3]]: memref<8x3xf32>) -> (memref<8x3xf32>, memref<8x3xf32>) {
// RANK2-BANKDIM1:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// RANK2-BANKDIM1:           %[[VAL_5:.*]] = memref.alloc() : memref<8x3xf32>
// RANK2-BANKDIM1:           %[[VAL_6:.*]] = memref.alloc() : memref<8x3xf32>
// RANK2-BANKDIM1:           affine.parallel (%[[VAL_7:.*]]) = (0) to (8) {
// RANK2-BANKDIM1:             affine.parallel (%[[VAL_8:.*]]) = (0) to (6) {
// RANK2-BANKDIM1:               %[[VAL_9:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]], %[[VAL_8]])
// RANK2-BANKDIM1:               %[[VAL_10:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]], %[[VAL_8]])
// RANK2-BANKDIM1:               %[[VAL_11:.*]] = scf.index_switch %[[VAL_9]] -> f32
// RANK2-BANKDIM1:               case 0 {
// RANK2-BANKDIM1:                 %[[VAL_12:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_10]]] : memref<8x3xf32>
// RANK2-BANKDIM1:                 scf.yield %[[VAL_12]] : f32
// RANK2-BANKDIM1:               }
// RANK2-BANKDIM1:               case 1 {
// RANK2-BANKDIM1:                 %[[VAL_13:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_7]], %[[VAL_10]]] : memref<8x3xf32>
// RANK2-BANKDIM1:                 scf.yield %[[VAL_13]] : f32
// RANK2-BANKDIM1:               }
// RANK2-BANKDIM1:               default {
// RANK2-BANKDIM1:                 scf.yield %[[VAL_4]] : f32
// RANK2-BANKDIM1:               }
// RANK2-BANKDIM1:               %[[VAL_14:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]], %[[VAL_8]])
// RANK2-BANKDIM1:               %[[VAL_15:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]], %[[VAL_8]])
// RANK2-BANKDIM1:               %[[VAL_16:.*]] = scf.index_switch %[[VAL_14]] -> f32
// RANK2-BANKDIM1:               case 0 {
// RANK2-BANKDIM1:                 %[[VAL_17:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_7]], %[[VAL_15]]] : memref<8x3xf32>
// RANK2-BANKDIM1:                 scf.yield %[[VAL_17]] : f32
// RANK2-BANKDIM1:               }
// RANK2-BANKDIM1:               case 1 {
// RANK2-BANKDIM1:                 %[[VAL_18:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_7]], %[[VAL_15]]] : memref<8x3xf32>
// RANK2-BANKDIM1:                 scf.yield %[[VAL_18]] : f32
// RANK2-BANKDIM1:               }
// RANK2-BANKDIM1:               default {
// RANK2-BANKDIM1:                 scf.yield %[[VAL_4]] : f32
// RANK2-BANKDIM1:               }
// RANK2-BANKDIM1:               %[[VAL_19:.*]] = arith.mulf %[[VAL_11]], %[[VAL_16]] : f32
// RANK2-BANKDIM1:               %[[VAL_20:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]], %[[VAL_8]])
// RANK2-BANKDIM1:               %[[VAL_21:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]], %[[VAL_8]])
// RANK2-BANKDIM1:               scf.index_switch %[[VAL_20]]
// RANK2-BANKDIM1:               case 0 {
// RANK2-BANKDIM1:                 affine.store %[[VAL_19]], %[[VAL_5]]{{\[}}%[[VAL_7]], %[[VAL_21]]] : memref<8x3xf32>
// RANK2-BANKDIM1:                 scf.yield
// RANK2-BANKDIM1:               }
// RANK2-BANKDIM1:               case 1 {
// RANK2-BANKDIM1:                 affine.store %[[VAL_19]], %[[VAL_6]]{{\[}}%[[VAL_7]], %[[VAL_21]]] : memref<8x3xf32>
// RANK2-BANKDIM1:                 scf.yield
// RANK2-BANKDIM1:               }
// RANK2-BANKDIM1:               default {
// RANK2-BANKDIM1:               }
// RANK2-BANKDIM1:             }
// RANK2-BANKDIM1:           }
// RANK2-BANKDIM1:           return %[[VAL_5]], %[[VAL_6]] : memref<8x3xf32>, memref<8x3xf32>
// RANK2-BANKDIM1:         }

func.func @rank_two_bank_dim1(%arg0: memref<8x6xf32>, %arg1: memref<8x6xf32>) -> (memref<8x6xf32>) {
  %mem = memref.alloc() : memref<8x6xf32>
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

