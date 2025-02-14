// RUN: circt-opt %s -memory-banking="factor=3 dimension=1"

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d1 mod 5)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d1 floordiv 5)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0 mod 5)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0 floordiv 5)>

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:arg0]]: memref<3x1xf32>,
// CHECK-SAME:                    %[[VAL_1:arg1]]: memref<3x1xf32>,
// CHECK-SAME:                    %[[VAL_2:arg2]]: memref<3x1xf32>,
// CHECK-SAME:                    %[[VAL_3:arg3]]: memref<3x1xf32>,
// CHECK-SAME:                    %[[VAL_4:arg4]]: memref<3x1xf32>,
// CHECK-SAME:                    %[[VAL_5:arg5]]: memref<1x3xf32>,
// CHECK-SAME:                    %[[VAL_6:arg6]]: memref<1x3xf32>,
// CHECK-SAME:                    %[[VAL_7:arg7]]: memref<1x3xf32>,
// CHECK-SAME:                    %[[VAL_8:arg8]]: memref<1x3xf32>,
// CHECK-SAME:                    %[[VAL_9:arg9]]: memref<1x3xf32>) -> (memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>) {
// CHECK:           %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<1x3xf32>
// CHECK:           %[[VAL_12:.*]] = memref.alloc() : memref<1x3xf32>
// CHECK:           %[[VAL_13:.*]] = memref.alloc() : memref<1x3xf32>
// CHECK:           %[[VAL_14:.*]] = memref.alloc() : memref<1x3xf32>
// CHECK:           %[[VAL_15:.*]] = memref.alloc() : memref<1x3xf32>
// CHECK:           affine.parallel (%[[VAL_16:.*]]) = (0) to (5) {
// CHECK:             affine.parallel (%[[VAL_17:.*]]) = (0) to (3) {
// CHECK:               %[[VAL_18:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_17]], %[[VAL_16]])
// CHECK:               %[[VAL_19:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_17]], %[[VAL_16]])
// CHECK:               %[[VAL_20:.*]] = scf.index_switch %[[VAL_18]] -> f32
// CHECK:               case 0 {
// CHECK:                 %[[VAL_21:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// CHECK:                 scf.yield %[[VAL_21]] : f32
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 %[[VAL_22:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// CHECK:                 scf.yield %[[VAL_22]] : f32
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 %[[VAL_23:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// CHECK:                 scf.yield %[[VAL_23]] : f32
// CHECK:               }
// CHECK:               case 3 {
// CHECK:                 %[[VAL_24:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// CHECK:                 scf.yield %[[VAL_24]] : f32
// CHECK:               }
// CHECK:               case 4 {
// CHECK:                 %[[VAL_25:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<3x1xf32>
// CHECK:                 scf.yield %[[VAL_25]] : f32
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_10]] : f32
// CHECK:               }
// CHECK:               %[[VAL_26:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_16]], %[[VAL_17]])
// CHECK:               %[[VAL_27:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_16]], %[[VAL_17]])
// CHECK:               %[[VAL_28:.*]] = scf.index_switch %[[VAL_26]] -> f32
// CHECK:               case 0 {
// CHECK:                 %[[VAL_29:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield %[[VAL_29]] : f32
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 %[[VAL_30:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield %[[VAL_30]] : f32
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 %[[VAL_31:.*]] = affine.load %[[VAL_7]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield %[[VAL_31]] : f32
// CHECK:               }
// CHECK:               case 3 {
// CHECK:                 %[[VAL_32:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield %[[VAL_32]] : f32
// CHECK:               }
// CHECK:               case 4 {
// CHECK:                 %[[VAL_33:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_27]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield %[[VAL_33]] : f32
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_10]] : f32
// CHECK:               }
// CHECK:               %[[VAL_34:.*]] = arith.mulf %[[VAL_20]], %[[VAL_28]] : f32
// CHECK:               %[[VAL_35:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_16]], %[[VAL_17]])
// CHECK:               %[[VAL_36:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_16]], %[[VAL_17]])
// CHECK:               scf.index_switch %[[VAL_35]]
// CHECK:               case 0 {
// CHECK:                 affine.store %[[VAL_34]], %[[VAL_11]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 affine.store %[[VAL_34]], %[[VAL_12]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 affine.store %[[VAL_34]], %[[VAL_13]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 3 {
// CHECK:                 affine.store %[[VAL_34]], %[[VAL_14]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 4 {
// CHECK:                 affine.store %[[VAL_34]], %[[VAL_15]]{{\[}}%[[VAL_36]], %[[VAL_17]]] : memref<1x3xf32>
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               default {
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %[[VAL_15]] : memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>, memref<1x3xf32>
// CHECK:         }

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
