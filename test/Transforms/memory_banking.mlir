// RUN: circt-opt %s -split-input-file -memory-banking="factors=2" | FileCheck %s --check-prefix UNROLL-BY-2
// RUN: circt-opt %s -split-input-file -memory-banking="factors=1" | FileCheck %s --check-prefix UNROLL-BY-1
// RUN: circt-opt %s -split-input-file -memory-banking="factors=8" | FileCheck %s --check-prefix UNROLL-BY-8
// RUN: circt-opt %s -split-input-file -memory-banking="factors=2" | FileCheck %s --check-prefix ALLOC-UNROLL-2

// -----

// UNROLL-BY-2: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 mod 2)>
// UNROLL-BY-2: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0 floordiv 2)>

// UNROLL-BY-2-LABEL:   func.func @bank_one_dim_unroll2(
// UNROLL-BY-2-SAME:                                %[[VAL_0:arg0]]: memref<4xf32>,
// UNROLL-BY-2-SAME:                                %[[VAL_1:arg1]]: memref<4xf32>,
// UNROLL-BY-2-SAME:                                %[[VAL_2:arg2]]: memref<4xf32>,
// UNROLL-BY-2-SAME:                                %[[VAL_3:arg3]]: memref<4xf32>) -> (memref<4xf32>, memref<4xf32>) {
// UNROLL-BY-2:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// UNROLL-BY-2:           %[[VAL_5:.*]] = memref.alloc() : memref<4xf32>
// UNROLL-BY-2:           %[[VAL_6:.*]] = memref.alloc() : memref<4xf32>
// UNROLL-BY-2:           affine.parallel (%[[VAL_7:.*]]) = (0) to (8) {
// UNROLL-BY-2:             %[[VAL_8:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]])
// UNROLL-BY-2:             %[[VAL_9:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]])
// UNROLL-BY-2:             %[[VAL_10:.*]] = scf.index_switch %[[VAL_8]] -> f32
// UNROLL-BY-2:             case 0 {
// UNROLL-BY-2:               %[[VAL_11:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_9]]] : memref<4xf32>
// UNROLL-BY-2:               scf.yield %[[VAL_11]] : f32
// UNROLL-BY-2:             }
// UNROLL-BY-2:             case 1 {
// UNROLL-BY-2:               %[[VAL_12:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_9]]] : memref<4xf32>
// UNROLL-BY-2:               scf.yield %[[VAL_12]] : f32
// UNROLL-BY-2:             }
// UNROLL-BY-2:             default {
// UNROLL-BY-2:               scf.yield %[[VAL_4]] : f32
// UNROLL-BY-2:             }
// UNROLL-BY-2:             %[[VAL_13:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]])
// UNROLL-BY-2:             %[[VAL_14:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]])
// UNROLL-BY-2:             %[[VAL_15:.*]] = scf.index_switch %[[VAL_13]] -> f32
// UNROLL-BY-2:             case 0 {
// UNROLL-BY-2:               %[[VAL_16:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_14]]] : memref<4xf32>
// UNROLL-BY-2:               scf.yield %[[VAL_16]] : f32
// UNROLL-BY-2:             }
// UNROLL-BY-2:             case 1 {
// UNROLL-BY-2:               %[[VAL_17:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_14]]] : memref<4xf32>
// UNROLL-BY-2:               scf.yield %[[VAL_17]] : f32
// UNROLL-BY-2:             }
// UNROLL-BY-2:             default {
// UNROLL-BY-2:               scf.yield %[[VAL_4]] : f32
// UNROLL-BY-2:             }
// UNROLL-BY-2:             %[[VAL_18:.*]] = arith.mulf %[[VAL_10]], %[[VAL_15]] : f32
// UNROLL-BY-2:             %[[VAL_19:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]])
// UNROLL-BY-2:             %[[VAL_20:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]])
// UNROLL-BY-2:             scf.index_switch %[[VAL_19]]
// UNROLL-BY-2:             case 0 {
// UNROLL-BY-2:               affine.store %[[VAL_18]], %[[VAL_5]]{{\[}}%[[VAL_20]]] : memref<4xf32>
// UNROLL-BY-2:               scf.yield
// UNROLL-BY-2:             }
// UNROLL-BY-2:             case 1 {
// UNROLL-BY-2:               affine.store %[[VAL_18]], %[[VAL_6]]{{\[}}%[[VAL_20]]] : memref<4xf32>
// UNROLL-BY-2:               scf.yield
// UNROLL-BY-2:             }
// UNROLL-BY-2:             default {
// UNROLL-BY-2:             }
// UNROLL-BY-2:           }
// UNROLL-BY-2:           return %[[VAL_5]], %[[VAL_6]] : memref<4xf32>, memref<4xf32>
// UNROLL-BY-2:         }

func.func @bank_one_dim_unroll2(%arg0: memref<8xf32>, %arg1: memref<8xf32>) -> (memref<8xf32>) {
  %mem = memref.alloc() : memref<8xf32>
  affine.parallel (%i) = (0) to (8) {
    %1 = affine.load %arg0[%i] : memref<8xf32>
    %2 = affine.load %arg1[%i] : memref<8xf32>
    %3 = arith.mulf %1, %2 : f32
    affine.store %3, %mem[%i] : memref<8xf32>
  }
  return %mem : memref<8xf32>
}

// -----

// UNROLL-BY-1-LABEL:   func.func @bank_one_dim_unroll1(
// UNROLL-BY-1-SAME:                                    %[[VAL_0:.*]]: memref<8xf32>,
// UNROLL-BY-1-SAME:                                    %[[VAL_1:.*]]: memref<8xf32>) -> memref<8xf32> {
// UNROLL-BY-1:           %[[VAL_2:.*]] = memref.alloc() : memref<8xf32>
// UNROLL-BY-1:           affine.parallel (%[[VAL_3:.*]]) = (0) to (8) {
// UNROLL-BY-1:             %[[VAL_4:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_3]]] : memref<8xf32>
// UNROLL-BY-1:             %[[VAL_5:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_3]]] : memref<8xf32>
// UNROLL-BY-1:             %[[VAL_6:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// UNROLL-BY-1:             affine.store %[[VAL_6]], %[[VAL_2]]{{\[}}%[[VAL_3]]] : memref<8xf32>
// UNROLL-BY-1:           }
// UNROLL-BY-1:           return %[[VAL_2]] : memref<8xf32>
// UNROLL-BY-1:         }

func.func @bank_one_dim_unroll1(%arg0: memref<8xf32>, %arg1: memref<8xf32>) -> (memref<8xf32>) {
  %mem = memref.alloc() : memref<8xf32>
  affine.parallel (%i) = (0) to (8) {
    %1 = affine.load %arg0[%i] : memref<8xf32>
    %2 = affine.load %arg1[%i] : memref<8xf32>
    %3 = arith.mulf %1, %2 : f32
    affine.store %3, %mem[%i] : memref<8xf32>
  }
  return %mem : memref<8xf32>
}

// -----

// UNROLL-BY-8: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 mod 8)>
// UNROLL-BY-8: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0 floordiv 8)>

// UNROLL-BY-8-LABEL:   func.func @bank_one_dim_unroll8(
// UNROLL-BY-8-SAME:                                    %[[VAL_0:arg0]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_1:arg1]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_2:arg2]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_3:arg3]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_4:arg4]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_5:arg5]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_6:arg6]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_7:arg7]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_8:arg8]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_9:arg9]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_10:arg10]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_11:arg11]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_12:arg12]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_13:arg13]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_14:arg14]]: memref<1xf32>,
// UNROLL-BY-8-SAME:                                    %[[VAL_15:arg15]]: memref<1xf32>) -> (memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) {
// UNROLL-BY-8:           %[[VAL_16:.*]] = arith.constant 0.000000e+00 : f32
// UNROLL-BY-8:           %[[VAL_17:.*]] = memref.alloc() : memref<1xf32>
// UNROLL-BY-8:           %[[VAL_18:.*]] = memref.alloc() : memref<1xf32>
// UNROLL-BY-8:           %[[VAL_19:.*]] = memref.alloc() : memref<1xf32>
// UNROLL-BY-8:           %[[VAL_20:.*]] = memref.alloc() : memref<1xf32>
// UNROLL-BY-8:           %[[VAL_21:.*]] = memref.alloc() : memref<1xf32>
// UNROLL-BY-8:           %[[VAL_22:.*]] = memref.alloc() : memref<1xf32>
// UNROLL-BY-8:           %[[VAL_23:.*]] = memref.alloc() : memref<1xf32>
// UNROLL-BY-8:           %[[VAL_24:.*]] = memref.alloc() : memref<1xf32>
// UNROLL-BY-8:           affine.parallel (%[[VAL_25:.*]]) = (0) to (8) {
// UNROLL-BY-8:             %[[VAL_26:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_25]])
// UNROLL-BY-8:             %[[VAL_27:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_25]])
// UNROLL-BY-8:             %[[VAL_28:.*]] = scf.index_switch %[[VAL_26]] -> f32
// UNROLL-BY-8:             case 0 {
// UNROLL-BY-8:               %[[VAL_29:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_27]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_29]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 1 {
// UNROLL-BY-8:               %[[VAL_30:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_27]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_30]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 2 {
// UNROLL-BY-8:               %[[VAL_31:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_27]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_31]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 3 {
// UNROLL-BY-8:               %[[VAL_32:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_27]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_32]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 4 {
// UNROLL-BY-8:               %[[VAL_33:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_27]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_33]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 5 {
// UNROLL-BY-8:               %[[VAL_34:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_27]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_34]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 6 {
// UNROLL-BY-8:               %[[VAL_35:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_27]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_35]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 7 {
// UNROLL-BY-8:               %[[VAL_36:.*]] = affine.load %[[VAL_7]]{{\[}}%[[VAL_27]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_36]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             default {
// UNROLL-BY-8:               scf.yield %[[VAL_16]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             %[[VAL_37:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_25]])
// UNROLL-BY-8:             %[[VAL_38:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_25]])
// UNROLL-BY-8:             %[[VAL_39:.*]] = scf.index_switch %[[VAL_37]] -> f32
// UNROLL-BY-8:             case 0 {
// UNROLL-BY-8:               %[[VAL_40:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_38]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_40]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 1 {
// UNROLL-BY-8:               %[[VAL_41:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_38]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_41]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 2 {
// UNROLL-BY-8:               %[[VAL_42:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_38]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_42]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 3 {
// UNROLL-BY-8:               %[[VAL_43:.*]] = affine.load %[[VAL_11]]{{\[}}%[[VAL_38]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_43]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 4 {
// UNROLL-BY-8:               %[[VAL_44:.*]] = affine.load %[[VAL_12]]{{\[}}%[[VAL_38]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_44]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 5 {
// UNROLL-BY-8:               %[[VAL_45:.*]] = affine.load %[[VAL_13]]{{\[}}%[[VAL_38]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_45]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 6 {
// UNROLL-BY-8:               %[[VAL_46:.*]] = affine.load %[[VAL_14]]{{\[}}%[[VAL_38]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_46]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 7 {
// UNROLL-BY-8:               %[[VAL_47:.*]] = affine.load %[[VAL_15]]{{\[}}%[[VAL_38]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield %[[VAL_47]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             default {
// UNROLL-BY-8:               scf.yield %[[VAL_16]] : f32
// UNROLL-BY-8:             }
// UNROLL-BY-8:             %[[VAL_48:.*]] = arith.mulf %[[VAL_28]], %[[VAL_39]] : f32
// UNROLL-BY-8:             %[[VAL_49:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_25]])
// UNROLL-BY-8:             %[[VAL_50:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_25]])
// UNROLL-BY-8:             scf.index_switch %[[VAL_49]]
// UNROLL-BY-8:             case 0 {
// UNROLL-BY-8:               affine.store %[[VAL_48]], %[[VAL_17]]{{\[}}%[[VAL_50]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 1 {
// UNROLL-BY-8:               affine.store %[[VAL_48]], %[[VAL_18]]{{\[}}%[[VAL_50]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 2 {
// UNROLL-BY-8:               affine.store %[[VAL_48]], %[[VAL_19]]{{\[}}%[[VAL_50]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 3 {
// UNROLL-BY-8:               affine.store %[[VAL_48]], %[[VAL_20]]{{\[}}%[[VAL_50]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 4 {
// UNROLL-BY-8:               affine.store %[[VAL_48]], %[[VAL_21]]{{\[}}%[[VAL_50]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 5 {
// UNROLL-BY-8:               affine.store %[[VAL_48]], %[[VAL_22]]{{\[}}%[[VAL_50]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 6 {
// UNROLL-BY-8:               affine.store %[[VAL_48]], %[[VAL_23]]{{\[}}%[[VAL_50]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield
// UNROLL-BY-8:             }
// UNROLL-BY-8:             case 7 {
// UNROLL-BY-8:               affine.store %[[VAL_48]], %[[VAL_24]]{{\[}}%[[VAL_50]]] : memref<1xf32>
// UNROLL-BY-8:               scf.yield
// UNROLL-BY-8:             }
// UNROLL-BY-8:             default {
// UNROLL-BY-8:             }
// UNROLL-BY-8:           }
// UNROLL-BY-8:           return %[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]] : memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>
// UNROLL-BY-8:         }

func.func @bank_one_dim_unroll8(%arg0: memref<8xf32>, %arg1: memref<8xf32>) -> (memref<8xf32>) {
  %mem = memref.alloc() : memref<8xf32>
  affine.parallel (%i) = (0) to (8) {
    %1 = affine.load %arg0[%i] : memref<8xf32>
    %2 = affine.load %arg1[%i] : memref<8xf32>
    %3 = arith.mulf %1, %2 : f32
    affine.store %3, %mem[%i] : memref<8xf32>
  }
  return %mem : memref<8xf32>
}

// -----

// ALLOC-UNROLL-2: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 mod 2)>
// ALLOC-UNROLL-2: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0 floordiv 2)>


// ALLOC-UNROLL-2-LABEL:   func.func @alloc_unroll2() -> (memref<4xf32>, memref<4xf32>) {
// ALLOC-UNROLL-2:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f32
// ALLOC-UNROLL-2:           %[[VAL_1:.*]] = memref.alloc() : memref<4xf32>
// ALLOC-UNROLL-2:           %[[VAL_2:.*]] = memref.alloc() : memref<4xf32>
// ALLOC-UNROLL-2:           %[[VAL_3:.*]] = memref.alloc() : memref<4xf32>
// ALLOC-UNROLL-2:           %[[VAL_4:.*]] = memref.alloc() : memref<4xf32>
// ALLOC-UNROLL-2:           %[[VAL_5:.*]] = memref.alloc() : memref<4xf32>
// ALLOC-UNROLL-2:           %[[VAL_6:.*]] = memref.alloc() : memref<4xf32>
// ALLOC-UNROLL-2:           affine.parallel (%[[VAL_7:.*]]) = (0) to (8) {
// ALLOC-UNROLL-2:             %[[VAL_8:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]])
// ALLOC-UNROLL-2:             %[[VAL_9:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]])
// ALLOC-UNROLL-2:             %[[VAL_10:.*]] = scf.index_switch %[[VAL_8]] -> f32
// ALLOC-UNROLL-2:             case 0 {
// ALLOC-UNROLL-2:               %[[VAL_11:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_9]]] : memref<4xf32>
// ALLOC-UNROLL-2:               scf.yield %[[VAL_11]] : f32
// ALLOC-UNROLL-2:             }
// ALLOC-UNROLL-2:             case 1 {
// ALLOC-UNROLL-2:               %[[VAL_12:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_9]]] : memref<4xf32>
// ALLOC-UNROLL-2:               scf.yield %[[VAL_12]] : f32
// ALLOC-UNROLL-2:             }
// ALLOC-UNROLL-2:             default {
// ALLOC-UNROLL-2:               scf.yield %[[VAL_0]] : f32
// ALLOC-UNROLL-2:             }
// ALLOC-UNROLL-2:             %[[VAL_13:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]])
// ALLOC-UNROLL-2:             %[[VAL_14:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]])
// ALLOC-UNROLL-2:             %[[VAL_15:.*]] = scf.index_switch %[[VAL_13]] -> f32
// ALLOC-UNROLL-2:             case 0 {
// ALLOC-UNROLL-2:               %[[VAL_16:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_14]]] : memref<4xf32>
// ALLOC-UNROLL-2:               scf.yield %[[VAL_16]] : f32
// ALLOC-UNROLL-2:             }
// ALLOC-UNROLL-2:             case 1 {
// ALLOC-UNROLL-2:               %[[VAL_17:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_14]]] : memref<4xf32>
// ALLOC-UNROLL-2:               scf.yield %[[VAL_17]] : f32
// ALLOC-UNROLL-2:             }
// ALLOC-UNROLL-2:             default {
// ALLOC-UNROLL-2:               scf.yield %[[VAL_0]] : f32
// ALLOC-UNROLL-2:             }
// ALLOC-UNROLL-2:             %[[VAL_18:.*]] = arith.mulf %[[VAL_10]], %[[VAL_15]] : f32
// ALLOC-UNROLL-2:             %[[VAL_19:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_7]])
// ALLOC-UNROLL-2:             %[[VAL_20:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_7]])
// ALLOC-UNROLL-2:             scf.index_switch %[[VAL_19]]
// ALLOC-UNROLL-2:             case 0 {
// ALLOC-UNROLL-2:               affine.store %[[VAL_18]], %[[VAL_5]]{{\[}}%[[VAL_20]]] : memref<4xf32>
// ALLOC-UNROLL-2:               scf.yield
// ALLOC-UNROLL-2:             }
// ALLOC-UNROLL-2:             case 1 {
// ALLOC-UNROLL-2:               affine.store %[[VAL_18]], %[[VAL_6]]{{\[}}%[[VAL_20]]] : memref<4xf32>
// ALLOC-UNROLL-2:               scf.yield
// ALLOC-UNROLL-2:             }
// ALLOC-UNROLL-2:             default {
// ALLOC-UNROLL-2:             }
// ALLOC-UNROLL-2:           }
// ALLOC-UNROLL-2:           return %[[VAL_5]], %[[VAL_6]] : memref<4xf32>, memref<4xf32>
// ALLOC-UNROLL-2:         }

func.func @alloc_unroll2() -> (memref<8xf32>) {
  %arg0 = memref.alloc() : memref<8xf32>
  %arg1 = memref.alloc() : memref<8xf32>
  %mem = memref.alloc() : memref<8xf32>
  affine.parallel (%i) = (0) to (8) {
    %1 = affine.load %arg0[%i] : memref<8xf32>
    %2 = affine.load %arg1[%i] : memref<8xf32>
    %3 = arith.mulf %1, %2 : f32
    affine.store %3, %mem[%i] : memref<8xf32>
  }
  return %mem : memref<8xf32>
}

