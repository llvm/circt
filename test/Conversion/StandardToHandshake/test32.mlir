// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @test(
// CHECK-SAME:                         %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:5 = memory[ld = 2, st = 1] (%[[VAL_2:.*]], %[[VAL_3:.*]], %[[VAL_4:.*]], %[[VAL_5:.*]]) {id = 0 : i32, lsq = false} : memref<10xf32>, (f32, index, index, index) -> (f32, f32, none, none, none)
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_1]]#4 : none
// CHECK:           %[[VAL_7:.*]]:2 = fork [2] %[[VAL_0]] : none
// CHECK:           %[[VAL_8:.*]]:3 = fork [3] %[[VAL_7]]#1 : none
// CHECK:           %[[VAL_9:.*]] = join %[[VAL_8]]#2, %[[VAL_1]]#3 : none
// CHECK:           %[[VAL_10:.*]] = constant %[[VAL_8]]#1 {value = 0 : index} : index
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_8]]#0 {value = 10 : index} : index
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]], %[[VAL_4]] = load {{\[}}%[[VAL_12]]#0] %[[VAL_1]]#0, %[[VAL_7]]#0 : index, f32
// CHECK:           %[[VAL_14:.*]] = br %[[VAL_9]] : none
// CHECK:           %[[VAL_15:.*]] = br %[[VAL_10]] : index
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_12]]#1 : index
// CHECK:           %[[VAL_17:.*]] = br %[[VAL_13]] : f32
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = control_merge %[[VAL_14]] : none
// CHECK:           %[[VAL_20:.*]]:3 = fork [3] %[[VAL_19]] : index
// CHECK:           %[[VAL_21:.*]] = buffer [1] %[[VAL_22:.*]] {initValues = [0], sequential = true} : i1
// CHECK:           %[[VAL_23:.*]]:4 = fork [4] %[[VAL_21]] : i1
// CHECK:           %[[VAL_24:.*]] = mux %[[VAL_23]]#3 {{\[}}%[[VAL_18]], %[[VAL_25:.*]]] : i1, none
// CHECK:           %[[VAL_26:.*]] = mux %[[VAL_20]]#2 {{\[}}%[[VAL_16]]] : index, index
// CHECK:           %[[VAL_27:.*]] = mux %[[VAL_23]]#2 {{\[}}%[[VAL_26]], %[[VAL_28:.*]]] : i1, index
// CHECK:           %[[VAL_29:.*]]:2 = fork [2] %[[VAL_27]] : index
// CHECK:           %[[VAL_30:.*]] = mux %[[VAL_20]]#1 {{\[}}%[[VAL_17]]] : index, f32
// CHECK:           %[[VAL_31:.*]] = mux %[[VAL_23]]#1 {{\[}}%[[VAL_30]], %[[VAL_32:.*]]] : i1, f32
// CHECK:           %[[VAL_33:.*]] = mux %[[VAL_20]]#0 {{\[}}%[[VAL_15]]] : index, index
// CHECK:           %[[VAL_34:.*]] = mux %[[VAL_23]]#0 {{\[}}%[[VAL_33]], %[[VAL_35:.*]]] : i1, index
// CHECK:           %[[VAL_36:.*]]:2 = fork [2] %[[VAL_34]] : index
// CHECK:           %[[VAL_22]] = merge %[[VAL_37:.*]]#0 : i1
// CHECK:           %[[VAL_38:.*]] = arith.cmpi slt, %[[VAL_36]]#0, %[[VAL_29]]#0 : index
// CHECK:           %[[VAL_37]]:5 = fork [5] %[[VAL_38]] : i1
// CHECK:           %[[VAL_39:.*]], %[[VAL_40:.*]] = cond_br %[[VAL_37]]#4, %[[VAL_29]]#1 : index
// CHECK:           sink %[[VAL_40]] : index
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = cond_br %[[VAL_37]]#3, %[[VAL_31]] : f32
// CHECK:           sink %[[VAL_42]] : f32
// CHECK:           %[[VAL_43:.*]], %[[VAL_44:.*]] = cond_br %[[VAL_37]]#2, %[[VAL_24]] : none
// CHECK:           %[[VAL_45:.*]], %[[VAL_46:.*]] = cond_br %[[VAL_37]]#1, %[[VAL_36]]#1 : index
// CHECK:           sink %[[VAL_46]] : index
// CHECK:           %[[VAL_47:.*]] = merge %[[VAL_45]] : index
// CHECK:           %[[VAL_48:.*]] = merge %[[VAL_41]] : f32
// CHECK:           %[[VAL_49:.*]]:2 = fork [2] %[[VAL_48]] : f32
// CHECK:           %[[VAL_50:.*]] = merge %[[VAL_39]] : index
// CHECK:           %[[VAL_51:.*]], %[[VAL_52:.*]] = control_merge %[[VAL_43]] : none
// CHECK:           %[[VAL_53:.*]]:3 = fork [3] %[[VAL_51]] : none
// CHECK:           %[[VAL_54:.*]]:2 = fork [2] %[[VAL_53]]#2 : none
// CHECK:           %[[VAL_55:.*]] = join %[[VAL_54]]#1, %[[VAL_6]]#1, %[[VAL_1]]#2 : none
// CHECK:           sink %[[VAL_52]] : index
// CHECK:           %[[VAL_56:.*]] = constant %[[VAL_54]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_57:.*]] = arith.addi %[[VAL_47]], %[[VAL_56]] : index
// CHECK:           %[[VAL_58:.*]]:3 = fork [3] %[[VAL_57]] : index
// CHECK:           %[[VAL_59:.*]], %[[VAL_5]] = load {{\[}}%[[VAL_58]]#2] %[[VAL_1]]#1, %[[VAL_53]]#1 : index, f32
// CHECK:           %[[VAL_60:.*]] = arith.addf %[[VAL_49]]#1, %[[VAL_59]] : f32
// CHECK:           %[[VAL_61:.*]] = join %[[VAL_53]]#0, %[[VAL_6]]#0 : none
// CHECK:           %[[VAL_2]], %[[VAL_3]] = store {{\[}}%[[VAL_58]]#1] %[[VAL_60]], %[[VAL_61]] : index, f32
// CHECK:           %[[VAL_32]] = br %[[VAL_49]]#0 : f32
// CHECK:           %[[VAL_28]] = br %[[VAL_50]] : index
// CHECK:           %[[VAL_25]] = br %[[VAL_55]] : none
// CHECK:           %[[VAL_35]] = br %[[VAL_58]]#0 : index
// CHECK:           %[[VAL_62:.*]], %[[VAL_63:.*]] = control_merge %[[VAL_44]] : none
// CHECK:           sink %[[VAL_63]] : index
// CHECK:           return %[[VAL_62]] : none
// CHECK:         }
func @test() {
  %10 = memref.alloc() : memref<10xf32>
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %5 = memref.load %10[%c10] : memref<10xf32>
  br ^bb1(%c0 : index)
^bb1(%1: index):      // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi slt, %1, %c10 : index
  cond_br %2, ^bb2, ^bb3
^bb2: // pred: ^bb1
  %c1 = arith.constant 1 : index
  %3 = arith.addi %1, %c1 : index
  %7 = memref.load %10[%3] : memref<10xf32>
  %8 = arith.addf %5, %7 : f32
  memref.store %8, %10[%3] : memref<10xf32>
  br ^bb1(%3 : index)
^bb3: // pred: ^bb1
  return
}
