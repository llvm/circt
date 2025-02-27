// RUN: circt-opt %s -memory-banking="factors=4,6 dimensions=0,1" --canonicalize --split-input-file | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0 mod 2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0 mod 3)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0 mod 4)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (d0 mod 6)>

// CHECK-LABEL:   func.func @multi_config(
// CHECK-SAME:                            %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<4x2xf32>,
// CHECK-SAME:                            %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_13:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_14:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_15:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_16:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_17:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_18:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_19:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_20:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_21:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_22:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_23:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_24:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_25:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_26:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_27:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_28:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>,
// CHECK-SAME:                            %[[VAL_29:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<2x1xf32>) -> (memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>) {
// CHECK:           %[[VAL_30:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_31:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_32:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_33:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_34:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_35:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_36:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_37:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           %[[VAL_38:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:           affine.parallel (%[[VAL_39:.*]]) = (0) to (8) {
// CHECK:             affine.parallel (%[[VAL_40:.*]]) = (0) to (6) {
// CHECK:               %[[VAL_41:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_39]])
// CHECK:               %[[VAL_42:.*]] = scf.index_switch %[[VAL_41]] -> f32
// CHECK:               case 0 {
// CHECK:                 %[[VAL_43:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_40]])
// CHECK:                 %[[VAL_44:.*]] = scf.index_switch %[[VAL_43]] -> f32
// CHECK:                 case 0 {
// CHECK:                   %[[VAL_45:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_39]] floordiv 2, %[[VAL_40]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_45]] : f32
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   %[[VAL_46:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_39]] floordiv 2, %[[VAL_40]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_46]] : f32
// CHECK:                 }
// CHECK:                 case 2 {
// CHECK:                   %[[VAL_47:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_39]] floordiv 2, %[[VAL_40]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_47]] : f32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_30]] : f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_44]] : f32
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 %[[VAL_48:.*]] = affine.apply #[[$ATTR_1]](%[[VAL_40]])
// CHECK:                 %[[VAL_49:.*]] = scf.index_switch %[[VAL_48]] -> f32
// CHECK:                 case 0 {
// CHECK:                   %[[VAL_50:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_39]] floordiv 2, %[[VAL_40]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_50]] : f32
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   %[[VAL_51:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_39]] floordiv 2, %[[VAL_40]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_51]] : f32
// CHECK:                 }
// CHECK:                 case 2 {
// CHECK:                   %[[VAL_52:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_39]] floordiv 2, %[[VAL_40]] floordiv 3] : memref<4x2xf32>
// CHECK:                   scf.yield %[[VAL_52]] : f32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_30]] : f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_49]] : f32
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_30]] : f32
// CHECK:               }
// CHECK:               %[[VAL_53:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_39]])
// CHECK:               %[[VAL_54:.*]] = scf.index_switch %[[VAL_53]] -> f32
// CHECK:               case 0 {
// CHECK:                 %[[VAL_55:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_40]])
// CHECK:                 %[[VAL_56:.*]] = scf.index_switch %[[VAL_55]] -> f32
// CHECK:                 case 0 {
// CHECK:                   %[[VAL_57:.*]] = affine.load %[[VAL_6]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_57]] : f32
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   %[[VAL_58:.*]] = affine.load %[[VAL_7]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_58]] : f32
// CHECK:                 }
// CHECK:                 case 2 {
// CHECK:                   %[[VAL_59:.*]] = affine.load %[[VAL_8]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_59]] : f32
// CHECK:                 }
// CHECK:                 case 3 {
// CHECK:                   %[[VAL_60:.*]] = affine.load %[[VAL_9]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_60]] : f32
// CHECK:                 }
// CHECK:                 case 4 {
// CHECK:                   %[[VAL_61:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_61]] : f32
// CHECK:                 }
// CHECK:                 case 5 {
// CHECK:                   %[[VAL_62:.*]] = affine.load %[[VAL_11]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_62]] : f32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_30]] : f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_56]] : f32
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 %[[VAL_63:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_40]])
// CHECK:                 %[[VAL_64:.*]] = scf.index_switch %[[VAL_63]] -> f32
// CHECK:                 case 0 {
// CHECK:                   %[[VAL_65:.*]] = affine.load %[[VAL_12]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_65]] : f32
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   %[[VAL_66:.*]] = affine.load %[[VAL_13]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_66]] : f32
// CHECK:                 }
// CHECK:                 case 2 {
// CHECK:                   %[[VAL_67:.*]] = affine.load %[[VAL_14]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_67]] : f32
// CHECK:                 }
// CHECK:                 case 3 {
// CHECK:                   %[[VAL_68:.*]] = affine.load %[[VAL_15]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_68]] : f32
// CHECK:                 }
// CHECK:                 case 4 {
// CHECK:                   %[[VAL_69:.*]] = affine.load %[[VAL_16]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_69]] : f32
// CHECK:                 }
// CHECK:                 case 5 {
// CHECK:                   %[[VAL_70:.*]] = affine.load %[[VAL_17]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_70]] : f32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_30]] : f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_64]] : f32
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 %[[VAL_71:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_40]])
// CHECK:                 %[[VAL_72:.*]] = scf.index_switch %[[VAL_71]] -> f32
// CHECK:                 case 0 {
// CHECK:                   %[[VAL_73:.*]] = affine.load %[[VAL_18]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_73]] : f32
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   %[[VAL_74:.*]] = affine.load %[[VAL_19]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_74]] : f32
// CHECK:                 }
// CHECK:                 case 2 {
// CHECK:                   %[[VAL_75:.*]] = affine.load %[[VAL_20]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_75]] : f32
// CHECK:                 }
// CHECK:                 case 3 {
// CHECK:                   %[[VAL_76:.*]] = affine.load %[[VAL_21]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_76]] : f32
// CHECK:                 }
// CHECK:                 case 4 {
// CHECK:                   %[[VAL_77:.*]] = affine.load %[[VAL_22]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_77]] : f32
// CHECK:                 }
// CHECK:                 case 5 {
// CHECK:                   %[[VAL_78:.*]] = affine.load %[[VAL_23]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_78]] : f32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_30]] : f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_72]] : f32
// CHECK:               }
// CHECK:               case 3 {
// CHECK:                 %[[VAL_79:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_40]])
// CHECK:                 %[[VAL_80:.*]] = scf.index_switch %[[VAL_79]] -> f32
// CHECK:                 case 0 {
// CHECK:                   %[[VAL_81:.*]] = affine.load %[[VAL_24]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_81]] : f32
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   %[[VAL_82:.*]] = affine.load %[[VAL_25]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_82]] : f32
// CHECK:                 }
// CHECK:                 case 2 {
// CHECK:                   %[[VAL_83:.*]] = affine.load %[[VAL_26]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_83]] : f32
// CHECK:                 }
// CHECK:                 case 3 {
// CHECK:                   %[[VAL_84:.*]] = affine.load %[[VAL_27]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_84]] : f32
// CHECK:                 }
// CHECK:                 case 4 {
// CHECK:                   %[[VAL_85:.*]] = affine.load %[[VAL_28]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_85]] : f32
// CHECK:                 }
// CHECK:                 case 5 {
// CHECK:                   %[[VAL_86:.*]] = affine.load %[[VAL_29]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 6] : memref<2x1xf32>
// CHECK:                   scf.yield %[[VAL_86]] : f32
// CHECK:                 }
// CHECK:                 default {
// CHECK:                   scf.yield %[[VAL_30]] : f32
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_80]] : f32
// CHECK:               }
// CHECK:               default {
// CHECK:                 scf.yield %[[VAL_30]] : f32
// CHECK:               }
// CHECK:               %[[VAL_87:.*]] = arith.mulf %[[VAL_42]], %[[VAL_54]] : f32
// CHECK:               %[[VAL_88:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_39]])
// CHECK:               scf.index_switch %[[VAL_88]]
// CHECK:               case 0 {
// CHECK:                 %[[VAL_89:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_40]])
// CHECK:                 scf.index_switch %[[VAL_89]]
// CHECK:                 case 0 {
// CHECK:                   affine.store %[[VAL_87]], %[[VAL_31]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   affine.store %[[VAL_87]], %[[VAL_32]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 default {
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 1 {
// CHECK:                 %[[VAL_90:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_40]])
// CHECK:                 scf.index_switch %[[VAL_90]]
// CHECK:                 case 0 {
// CHECK:                   affine.store %[[VAL_87]], %[[VAL_33]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   affine.store %[[VAL_87]], %[[VAL_34]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 default {
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 2 {
// CHECK:                 %[[VAL_91:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_40]])
// CHECK:                 scf.index_switch %[[VAL_91]]
// CHECK:                 case 0 {
// CHECK:                   affine.store %[[VAL_87]], %[[VAL_35]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   affine.store %[[VAL_87]], %[[VAL_36]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 default {
// CHECK:                 }
// CHECK:                 scf.yield
// CHECK:               }
// CHECK:               case 3 {
// CHECK:                 %[[VAL_92:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_40]])
// CHECK:                 scf.index_switch %[[VAL_92]]
// CHECK:                 case 0 {
// CHECK:                   affine.store %[[VAL_87]], %[[VAL_37]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 2] : memref<2x3xf32>
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:                 case 1 {
// CHECK:                   affine.store %[[VAL_87]], %[[VAL_38]]{{\[}}%[[VAL_39]] floordiv 4, %[[VAL_40]] floordiv 2] : memref<2x3xf32>
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
// CHECK:           return %[[VAL_31]], %[[VAL_32]], %[[VAL_33]], %[[VAL_34]], %[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_38]] : memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>, memref<2x3xf32>
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
