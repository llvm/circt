// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @affine_load(
// CHECK-SAME:                                %[[VAL_0:.*]]: index,
// CHECK-SAME:                                %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]]:7 = memory[ld = 3, st = 1] (%[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]], %[[VAL_6:.*]], %[[VAL_7:.*]]) {id = 0 : i32, lsq = false} : memref<10xf32>, (f32, index, index, index, index) -> (f32, f32, f32, none, none, none, none)
// CHECK:           %[[VAL_8:.*]]:2 = fork [2] %[[VAL_2]]#6 : none
// CHECK:           %[[VAL_9:.*]]:2 = fork [2] %[[VAL_2]]#5 : none
// CHECK:           %[[VAL_10:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_11:.*]]:2 = fork [2] %[[VAL_1]] : none
// CHECK:           %[[VAL_12:.*]]:4 = fork [4] %[[VAL_11]]#1 : none
// CHECK:           %[[VAL_13:.*]] = join %[[VAL_12]]#3, %[[VAL_2]]#4 : none
// CHECK:           %[[VAL_14:.*]] = constant %[[VAL_12]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_15:.*]]:2 = fork [2] %[[VAL_14]] : index
// CHECK:           %[[VAL_16:.*]] = constant %[[VAL_12]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_17:.*]], %[[VAL_5]] = load {{\[}}%[[VAL_15]]#0] %[[VAL_2]]#0, %[[VAL_11]]#0 : index, f32
// CHECK:           %[[VAL_18:.*]] = constant %[[VAL_12]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_19:.*]] = br %[[VAL_10]] : index
// CHECK:           %[[VAL_20:.*]] = br %[[VAL_13]] : none
// CHECK:           %[[VAL_21:.*]] = br %[[VAL_15]]#1 : index
// CHECK:           %[[VAL_22:.*]] = br %[[VAL_16]] : index
// CHECK:           %[[VAL_23:.*]] = br %[[VAL_17]] : f32
// CHECK:           %[[VAL_24:.*]] = br %[[VAL_18]] : index
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = control_merge %[[VAL_20]] : none
// CHECK:           %[[VAL_27:.*]]:5 = fork [5] %[[VAL_26]] : index
// CHECK:           %[[VAL_28:.*]] = buffer [1] seq %[[VAL_29:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_30:.*]]:6 = fork [6] %[[VAL_28]] : i1
// CHECK:           %[[VAL_31:.*]] = mux %[[VAL_30]]#5 {{\[}}%[[VAL_25]], %[[VAL_32:.*]]] : i1, none
// CHECK:           %[[VAL_33:.*]] = mux %[[VAL_27]]#4 {{\[}}%[[VAL_22]]] : index, index
// CHECK:           %[[VAL_34:.*]] = mux %[[VAL_30]]#4 {{\[}}%[[VAL_33]], %[[VAL_35:.*]]] : i1, index
// CHECK:           %[[VAL_36:.*]]:2 = fork [2] %[[VAL_34]] : index
// CHECK:           %[[VAL_37:.*]] = mux %[[VAL_27]]#3 {{\[}}%[[VAL_19]]] : index, index
// CHECK:           %[[VAL_38:.*]] = mux %[[VAL_30]]#3 {{\[}}%[[VAL_37]], %[[VAL_39:.*]]] : i1, index
// CHECK:           %[[VAL_40:.*]] = mux %[[VAL_27]]#2 {{\[}}%[[VAL_24]]] : index, index
// CHECK:           %[[VAL_41:.*]] = mux %[[VAL_30]]#2 {{\[}}%[[VAL_40]], %[[VAL_42:.*]]] : i1, index
// CHECK:           %[[VAL_43:.*]] = mux %[[VAL_27]]#1 {{\[}}%[[VAL_23]]] : index, f32
// CHECK:           %[[VAL_44:.*]] = mux %[[VAL_30]]#1 {{\[}}%[[VAL_43]], %[[VAL_45:.*]]] : i1, f32
// CHECK:           %[[VAL_46:.*]] = mux %[[VAL_27]]#0 {{\[}}%[[VAL_21]]] : index, index
// CHECK:           %[[VAL_47:.*]] = mux %[[VAL_30]]#0 {{\[}}%[[VAL_46]], %[[VAL_48:.*]]] : i1, index
// CHECK:           %[[VAL_49:.*]]:2 = fork [2] %[[VAL_47]] : index
// CHECK:           %[[VAL_29]] = merge %[[VAL_50:.*]]#0 : i1
// CHECK:           %[[VAL_51:.*]] = arith.cmpi slt, %[[VAL_49]]#0, %[[VAL_36]]#0 : index
// CHECK:           %[[VAL_50]]:7 = fork [7] %[[VAL_51]] : i1
// CHECK:           %[[VAL_52:.*]], %[[VAL_53:.*]] = cond_br %[[VAL_50]]#6, %[[VAL_36]]#1 : index
// CHECK:           sink %[[VAL_53]] : index
// CHECK:           %[[VAL_54:.*]], %[[VAL_55:.*]] = cond_br %[[VAL_50]]#5, %[[VAL_38]] : index
// CHECK:           sink %[[VAL_55]] : index
// CHECK:           %[[VAL_56:.*]], %[[VAL_57:.*]] = cond_br %[[VAL_50]]#4, %[[VAL_41]] : index
// CHECK:           sink %[[VAL_57]] : index
// CHECK:           %[[VAL_58:.*]], %[[VAL_59:.*]] = cond_br %[[VAL_50]]#3, %[[VAL_44]] : f32
// CHECK:           sink %[[VAL_59]] : f32
// CHECK:           %[[VAL_60:.*]], %[[VAL_61:.*]] = cond_br %[[VAL_50]]#2, %[[VAL_31]] : none
// CHECK:           %[[VAL_62:.*]], %[[VAL_63:.*]] = cond_br %[[VAL_50]]#1, %[[VAL_49]]#1 : index
// CHECK:           sink %[[VAL_63]] : index
// CHECK:           %[[VAL_64:.*]] = merge %[[VAL_62]] : index
// CHECK:           %[[VAL_65:.*]]:2 = fork [2] %[[VAL_64]] : index
// CHECK:           %[[VAL_66:.*]] = merge %[[VAL_54]] : index
// CHECK:           %[[VAL_67:.*]]:2 = fork [2] %[[VAL_66]] : index
// CHECK:           %[[VAL_68:.*]] = merge %[[VAL_56]] : index
// CHECK:           %[[VAL_69:.*]]:2 = fork [2] %[[VAL_68]] : index
// CHECK:           %[[VAL_70:.*]] = merge %[[VAL_58]] : f32
// CHECK:           %[[VAL_71:.*]]:3 = fork [3] %[[VAL_70]] : f32
// CHECK:           %[[VAL_72:.*]] = merge %[[VAL_52]] : index
// CHECK:           %[[VAL_73:.*]], %[[VAL_74:.*]] = control_merge %[[VAL_60]] : none
// CHECK:           %[[VAL_75:.*]]:4 = fork [4] %[[VAL_73]] : none
// CHECK:           %[[VAL_76:.*]]:2 = fork [2] %[[VAL_75]]#3 : none
// CHECK:           %[[VAL_77:.*]] = join %[[VAL_76]]#1, %[[VAL_9]]#1, %[[VAL_8]]#1, %[[VAL_2]]#3 : none
// CHECK:           sink %[[VAL_74]] : index
// CHECK:           %[[VAL_78:.*]] = arith.addi %[[VAL_65]]#1, %[[VAL_67]]#1 : index
// CHECK:           %[[VAL_79:.*]] = constant %[[VAL_76]]#0 {value = 7 : index} : index
// CHECK:           %[[VAL_80:.*]] = arith.addi %[[VAL_78]], %[[VAL_79]] : index
// CHECK:           %[[VAL_81:.*]]:3 = fork [3] %[[VAL_80]] : index
// CHECK:           %[[VAL_82:.*]], %[[VAL_6]] = load {{\[}}%[[VAL_81]]#2] %[[VAL_2]]#1, %[[VAL_75]]#2 : index, f32
// CHECK:           %[[VAL_83:.*]] = arith.addi %[[VAL_65]]#0, %[[VAL_69]]#1 : index
// CHECK:           %[[VAL_84:.*]], %[[VAL_7]] = load {{\[}}%[[VAL_81]]#1] %[[VAL_2]]#2, %[[VAL_75]]#1 : index, f32
// CHECK:           %[[VAL_85:.*]] = arith.addf %[[VAL_82]], %[[VAL_84]] : f32
// CHECK:           %[[VAL_86:.*]] = arith.addf %[[VAL_71]]#1, %[[VAL_71]]#2 : f32
// CHECK:           sink %[[VAL_86]] : f32
// CHECK:           %[[VAL_87:.*]] = join %[[VAL_75]]#0, %[[VAL_9]]#0, %[[VAL_8]]#0 : none
// CHECK:           %[[VAL_3]], %[[VAL_4]] = store {{\[}}%[[VAL_81]]#0] %[[VAL_85]], %[[VAL_87]] : index, f32
// CHECK:           %[[VAL_39]] = br %[[VAL_67]]#0 : index
// CHECK:           %[[VAL_42]] = br %[[VAL_69]]#0 : index
// CHECK:           %[[VAL_45]] = br %[[VAL_71]]#0 : f32
// CHECK:           %[[VAL_35]] = br %[[VAL_72]] : index
// CHECK:           %[[VAL_32]] = br %[[VAL_77]] : none
// CHECK:           %[[VAL_48]] = br %[[VAL_83]] : index
// CHECK:           %[[VAL_88:.*]], %[[VAL_89:.*]] = control_merge %[[VAL_61]] : none
// CHECK:           sink %[[VAL_89]] : index
// CHECK:           return %[[VAL_88]] : none
// CHECK:         }
func.func @affine_load(%arg0: index) {
  %0 = memref.alloc() : memref<10xf32>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %9 = memref.load %0[%c0] : memref<10xf32>
  %c1 = arith.constant 1 : index
  cf.br ^bb1(%c0 : index)
^bb1(%1: index):      // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi slt, %1, %c10 : index
  cf.cond_br %2, ^bb2, ^bb3
^bb2: // pred: ^bb1
  %3 = arith.addi %1, %arg0 : index
  %c7 = arith.constant 7 : index
  %4 = arith.addi %3, %c7 : index
  %5 = memref.load %0[%4] : memref<10xf32>
  %6 = arith.addi %1, %c1 : index
  %7 = memref.load %0[%4] : memref<10xf32>
  %8 = arith.addf %5, %7 : f32
  %11 = arith.addf %9, %9 : f32
  memref.store %8, %0[%4] : memref<10xf32>
  cf.br ^bb1(%6 : index)
^bb3: // pred: ^bb1
  return
}
