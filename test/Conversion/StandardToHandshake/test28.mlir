// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @affine_load(
// CHECK-SAME:                                %[[VAL_0:.*]]: index,
// CHECK-SAME:                                %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]]:3 = memory[ld = 1, st = 1] (%[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]]) {id = 1 : i32, lsq = false} : memref<10xf32>, (f32, index, index) -> (f32, none, none)
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_2]]#2 : none
// CHECK:           %[[VAL_7:.*]]:2 = memory[ld = 1, st = 0] (%[[VAL_8:.*]]) {id = 0 : i32, lsq = false} : memref<10xf32>, (index) -> (f32, none)
// CHECK:           %[[VAL_9:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_10:.*]]:4 = fork [4] %[[VAL_1]] : none
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_10]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_12:.*]] = constant %[[VAL_10]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_13:.*]] = constant %[[VAL_10]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_14:.*]] = br %[[VAL_9]] : index
// CHECK:           %[[VAL_15:.*]] = br %[[VAL_10]]#3 : none
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_11]] : index
// CHECK:           %[[VAL_17:.*]] = br %[[VAL_12]] : index
// CHECK:           %[[VAL_18:.*]] = br %[[VAL_13]] : index
// CHECK:           %[[VAL_19:.*]], %[[VAL_20:.*]] = control_merge %[[VAL_15]] : none
// CHECK:           %[[VAL_21:.*]]:4 = fork [4] %[[VAL_20]] : index
// CHECK:           %[[VAL_22:.*]] = buffer [1] seq %[[VAL_23:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_24:.*]]:5 = fork [5] %[[VAL_22]] : i1
// CHECK:           %[[VAL_25:.*]] = mux %[[VAL_24]]#4 {{\[}}%[[VAL_19]], %[[VAL_26:.*]]] : i1, none
// CHECK:           %[[VAL_27:.*]] = mux %[[VAL_21]]#3 {{\[}}%[[VAL_17]]] : index, index
// CHECK:           %[[VAL_28:.*]] = mux %[[VAL_24]]#3 {{\[}}%[[VAL_27]], %[[VAL_29:.*]]] : i1, index
// CHECK:           %[[VAL_30:.*]]:2 = fork [2] %[[VAL_28]] : index
// CHECK:           %[[VAL_31:.*]] = mux %[[VAL_21]]#2 {{\[}}%[[VAL_14]]] : index, index
// CHECK:           %[[VAL_32:.*]] = mux %[[VAL_24]]#2 {{\[}}%[[VAL_31]], %[[VAL_33:.*]]] : i1, index
// CHECK:           %[[VAL_34:.*]] = mux %[[VAL_21]]#1 {{\[}}%[[VAL_18]]] : index, index
// CHECK:           %[[VAL_35:.*]] = mux %[[VAL_24]]#1 {{\[}}%[[VAL_34]], %[[VAL_36:.*]]] : i1, index
// CHECK:           %[[VAL_37:.*]] = mux %[[VAL_21]]#0 {{\[}}%[[VAL_16]]] : index, index
// CHECK:           %[[VAL_38:.*]] = mux %[[VAL_24]]#0 {{\[}}%[[VAL_37]], %[[VAL_39:.*]]] : i1, index
// CHECK:           %[[VAL_40:.*]]:2 = fork [2] %[[VAL_38]] : index
// CHECK:           %[[VAL_23]] = merge %[[VAL_41:.*]]#0 : i1
// CHECK:           %[[VAL_42:.*]] = arith.cmpi slt, %[[VAL_40]]#0, %[[VAL_30]]#0 : index
// CHECK:           %[[VAL_41]]:6 = fork [6] %[[VAL_42]] : i1
// CHECK:           %[[VAL_43:.*]], %[[VAL_44:.*]] = cond_br %[[VAL_41]]#5, %[[VAL_30]]#1 : index
// CHECK:           sink %[[VAL_44]] : index
// CHECK:           %[[VAL_45:.*]], %[[VAL_46:.*]] = cond_br %[[VAL_41]]#4, %[[VAL_32]] : index
// CHECK:           sink %[[VAL_46]] : index
// CHECK:           %[[VAL_47:.*]], %[[VAL_48:.*]] = cond_br %[[VAL_41]]#3, %[[VAL_35]] : index
// CHECK:           sink %[[VAL_48]] : index
// CHECK:           %[[VAL_49:.*]], %[[VAL_50:.*]] = cond_br %[[VAL_41]]#2, %[[VAL_25]] : none
// CHECK:           %[[VAL_51:.*]], %[[VAL_52:.*]] = cond_br %[[VAL_41]]#1, %[[VAL_40]]#1 : index
// CHECK:           sink %[[VAL_52]] : index
// CHECK:           %[[VAL_53:.*]] = merge %[[VAL_51]] : index
// CHECK:           %[[VAL_54:.*]]:2 = fork [2] %[[VAL_53]] : index
// CHECK:           %[[VAL_55:.*]] = merge %[[VAL_45]] : index
// CHECK:           %[[VAL_56:.*]]:2 = fork [2] %[[VAL_55]] : index
// CHECK:           %[[VAL_57:.*]] = merge %[[VAL_47]] : index
// CHECK:           %[[VAL_58:.*]]:2 = fork [2] %[[VAL_57]] : index
// CHECK:           %[[VAL_59:.*]] = merge %[[VAL_43]] : index
// CHECK:           %[[VAL_60:.*]], %[[VAL_61:.*]] = control_merge %[[VAL_49]] : none
// CHECK:           %[[VAL_62:.*]]:4 = fork [4] %[[VAL_60]] : none
// CHECK:           %[[VAL_63:.*]]:2 = fork [2] %[[VAL_62]]#3 : none
// CHECK:           %[[VAL_64:.*]] = join %[[VAL_63]]#1, %[[VAL_7]]#1, %[[VAL_6]]#1, %[[VAL_2]]#1 : none
// CHECK:           sink %[[VAL_61]] : index
// CHECK:           %[[VAL_65:.*]] = arith.addi %[[VAL_54]]#1, %[[VAL_56]]#1 : index
// CHECK:           %[[VAL_66:.*]] = constant %[[VAL_63]]#0 {value = 7 : index} : index
// CHECK:           %[[VAL_67:.*]] = arith.addi %[[VAL_65]], %[[VAL_66]] : index
// CHECK:           %[[VAL_68:.*]]:3 = fork [3] %[[VAL_67]] : index
// CHECK:           %[[VAL_69:.*]], %[[VAL_8]] = load {{\[}}%[[VAL_68]]#2] %[[VAL_7]]#0, %[[VAL_62]]#2 : index, f32
// CHECK:           %[[VAL_70:.*]] = arith.addi %[[VAL_54]]#0, %[[VAL_58]]#1 : index
// CHECK:           %[[VAL_71:.*]], %[[VAL_5]] = load {{\[}}%[[VAL_68]]#1] %[[VAL_2]]#0, %[[VAL_62]]#1 : index, f32
// CHECK:           %[[VAL_72:.*]] = arith.addf %[[VAL_69]], %[[VAL_71]] : f32
// CHECK:           %[[VAL_73:.*]] = join %[[VAL_62]]#0, %[[VAL_6]]#0 : none
// CHECK:           %[[VAL_3]], %[[VAL_4]] = store {{\[}}%[[VAL_68]]#0] %[[VAL_72]], %[[VAL_73]] : index, f32
// CHECK:           %[[VAL_33]] = br %[[VAL_56]]#0 : index
// CHECK:           %[[VAL_36]] = br %[[VAL_58]]#0 : index
// CHECK:           %[[VAL_29]] = br %[[VAL_59]] : index
// CHECK:           %[[VAL_26]] = br %[[VAL_64]] : none
// CHECK:           %[[VAL_39]] = br %[[VAL_70]] : index
// CHECK:           %[[VAL_74:.*]], %[[VAL_75:.*]] = control_merge %[[VAL_50]] : none
// CHECK:           sink %[[VAL_75]] : index
// CHECK:           return %[[VAL_74]] : none
// CHECK:         }
func @affine_load(%arg0: index) {
  %0 = memref.alloc() : memref<10xf32>
  %10 = memref.alloc() : memref<10xf32>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
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
  %7 = memref.load %10[%4] : memref<10xf32>
  %8 = arith.addf %5, %7 : f32
  memref.store %8, %10[%4] : memref<10xf32>
  cf.br ^bb1(%6 : index)
^bb3: // pred: ^bb1
  return
}
