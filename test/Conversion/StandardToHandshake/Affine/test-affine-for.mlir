// RUN: circt-opt %s -lower-std-to-handshake -split-input-file | FileCheck %s

// -----

// Simple affine.for with an empty loop body.

// CHECK-LABEL:   handshake.func @empty_body(
// CHECK-SAME:                               %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:4 = fork [4] %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_1]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_1]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_1]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_5:.*]] = br %[[VAL_1]]#3 : none
// CHECK:           %[[VAL_6:.*]] = br %[[VAL_2]] : index
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_3]] : index
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_4]] : index
// CHECK:           %[[VAL_9:.*]] = mux %[[VAL_10:.*]]#2 {{\[}}%[[VAL_7]], %[[VAL_11:.*]]] : index, index
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_9]] : index
// CHECK:           %[[VAL_13:.*]] = mux %[[VAL_10]]#1 {{\[}}%[[VAL_8]], %[[VAL_14:.*]]] : index, index
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = control_merge %[[VAL_5]], %[[VAL_17:.*]] : none
// CHECK:           %[[VAL_10]]:3 = fork [3] %[[VAL_16]] : index
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_10]]#0 {{\[}}%[[VAL_6]], %[[VAL_19:.*]]] : index, index
// CHECK:           %[[VAL_20:.*]]:2 = fork [2] %[[VAL_18]] : index
// CHECK:           %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]]#1, %[[VAL_12]]#1 : index
// CHECK:           %[[VAL_22:.*]]:4 = fork [4] %[[VAL_21]] : i1
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = cond_br %[[VAL_22]]#3, %[[VAL_12]]#0 : index
// CHECK:           sink %[[VAL_24]] : index
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = cond_br %[[VAL_22]]#2, %[[VAL_13]] : index
// CHECK:           sink %[[VAL_26]] : index
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]] = cond_br %[[VAL_22]]#1, %[[VAL_15]] : none
// CHECK:           %[[VAL_29:.*]], %[[VAL_30:.*]] = cond_br %[[VAL_22]]#0, %[[VAL_20]]#0 : index
// CHECK:           sink %[[VAL_30]] : index
// CHECK:           %[[VAL_31:.*]] = merge %[[VAL_29]] : index
// CHECK:           %[[VAL_32:.*]] = merge %[[VAL_25]] : index
// CHECK:           %[[VAL_33:.*]] = merge %[[VAL_23]] : index
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = control_merge %[[VAL_27]] : none
// CHECK:           sink %[[VAL_35]] : index
// CHECK:           %[[VAL_36:.*]] = br %[[VAL_31]] : index
// CHECK:           %[[VAL_37:.*]] = br %[[VAL_32]] : index
// CHECK:           %[[VAL_38:.*]] = br %[[VAL_33]] : index
// CHECK:           %[[VAL_39:.*]] = br %[[VAL_34]] : none
// CHECK:           %[[VAL_40:.*]] = merge %[[VAL_36]] : index
// CHECK:           %[[VAL_41:.*]] = merge %[[VAL_37]] : index
// CHECK:           %[[VAL_42:.*]]:2 = fork [2] %[[VAL_41]] : index
// CHECK:           %[[VAL_43:.*]] = merge %[[VAL_38]] : index
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = control_merge %[[VAL_39]] : none
// CHECK:           sink %[[VAL_45]] : index
// CHECK:           %[[VAL_46:.*]] = arith.addi %[[VAL_40]], %[[VAL_42]]#1 : index
// CHECK:           %[[VAL_14]] = br %[[VAL_42]]#0 : index
// CHECK:           %[[VAL_11]] = br %[[VAL_43]] : index
// CHECK:           %[[VAL_17]] = br %[[VAL_44]] : none
// CHECK:           %[[VAL_19]] = br %[[VAL_46]] : index
// CHECK:           %[[VAL_47:.*]], %[[VAL_48:.*]] = control_merge %[[VAL_28]] : none
// CHECK:           sink %[[VAL_48]] : index
// CHECK:           return %[[VAL_47]] : none
// CHECK:         }
func @empty_body () -> () {
  affine.for %i = 0 to 10 { }
  return
}


// -----

// Simple load store pair in the loop body.

// CHECK-LABEL:   handshake.func @load_store(
// CHECK-SAME:                               %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:3 = memory[ld = 1, st = 1] (%[[VAL_2:.*]], %[[VAL_3:.*]], %[[VAL_4:.*]]) {id = 0 : i32, lsq = false} : memref<10xf32>, (f32, index, index) -> (f32, none, none)
// CHECK:           %[[VAL_5:.*]]:2 = fork [2] %[[VAL_1]]#2 : none
// CHECK:           %[[VAL_6:.*]]:4 = fork [4] %[[VAL_0]] : none
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_6]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_6]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_9:.*]] = constant %[[VAL_6]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_10:.*]] = br %[[VAL_6]]#3 : none
// CHECK:           %[[VAL_11:.*]] = br %[[VAL_7]] : index
// CHECK:           %[[VAL_12:.*]] = br %[[VAL_8]] : index
// CHECK:           %[[VAL_13:.*]] = br %[[VAL_9]] : index
// CHECK:           %[[VAL_14:.*]] = mux %[[VAL_15:.*]]#2 {{\[}}%[[VAL_12]], %[[VAL_16:.*]]] : index, index
// CHECK:           %[[VAL_17:.*]]:2 = fork [2] %[[VAL_14]] : index
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_15]]#1 {{\[}}%[[VAL_13]], %[[VAL_19:.*]]] : index, index
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = control_merge %[[VAL_10]], %[[VAL_22:.*]] : none
// CHECK:           %[[VAL_15]]:3 = fork [3] %[[VAL_21]] : index
// CHECK:           %[[VAL_23:.*]] = mux %[[VAL_15]]#0 {{\[}}%[[VAL_11]], %[[VAL_24:.*]]] : index, index
// CHECK:           %[[VAL_25:.*]]:2 = fork [2] %[[VAL_23]] : index
// CHECK:           %[[VAL_26:.*]] = arith.cmpi slt, %[[VAL_25]]#1, %[[VAL_17]]#1 : index
// CHECK:           %[[VAL_27:.*]]:4 = fork [4] %[[VAL_26]] : i1
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = cond_br %[[VAL_27]]#3, %[[VAL_17]]#0 : index
// CHECK:           sink %[[VAL_29]] : index
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = cond_br %[[VAL_27]]#2, %[[VAL_18]] : index
// CHECK:           sink %[[VAL_31]] : index
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = cond_br %[[VAL_27]]#1, %[[VAL_20]] : none
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = cond_br %[[VAL_27]]#0, %[[VAL_25]]#0 : index
// CHECK:           sink %[[VAL_35]] : index
// CHECK:           %[[VAL_36:.*]] = merge %[[VAL_34]] : index
// CHECK:           %[[VAL_37:.*]]:3 = fork [3] %[[VAL_36]] : index
// CHECK:           %[[VAL_38:.*]] = merge %[[VAL_30]] : index
// CHECK:           %[[VAL_39:.*]] = merge %[[VAL_28]] : index
// CHECK:           %[[VAL_40:.*]], %[[VAL_41:.*]] = control_merge %[[VAL_32]] : none
// CHECK:           %[[VAL_42:.*]]:3 = fork [3] %[[VAL_40]] : none
// CHECK:           %[[VAL_43:.*]] = join %[[VAL_42]]#2, %[[VAL_5]]#1, %[[VAL_1]]#1 : none
// CHECK:           sink %[[VAL_41]] : index
// CHECK:           %[[VAL_44:.*]], %[[VAL_4]] = load {{\[}}%[[VAL_37]]#2] %[[VAL_1]]#0, %[[VAL_42]]#1 : index, f32
// CHECK:           %[[VAL_45:.*]] = join %[[VAL_42]]#0, %[[VAL_5]]#0 : none
// CHECK:           %[[VAL_2]], %[[VAL_3]] = store {{\[}}%[[VAL_37]]#1] %[[VAL_44]], %[[VAL_45]] : index, f32
// CHECK:           %[[VAL_46:.*]] = br %[[VAL_37]]#0 : index
// CHECK:           %[[VAL_47:.*]] = br %[[VAL_38]] : index
// CHECK:           %[[VAL_48:.*]] = br %[[VAL_39]] : index
// CHECK:           %[[VAL_49:.*]] = br %[[VAL_43]] : none
// CHECK:           %[[VAL_50:.*]] = merge %[[VAL_46]] : index
// CHECK:           %[[VAL_51:.*]] = merge %[[VAL_47]] : index
// CHECK:           %[[VAL_52:.*]]:2 = fork [2] %[[VAL_51]] : index
// CHECK:           %[[VAL_53:.*]] = merge %[[VAL_48]] : index
// CHECK:           %[[VAL_54:.*]], %[[VAL_55:.*]] = control_merge %[[VAL_49]] : none
// CHECK:           sink %[[VAL_55]] : index
// CHECK:           %[[VAL_56:.*]] = arith.addi %[[VAL_50]], %[[VAL_52]]#1 : index
// CHECK:           %[[VAL_19]] = br %[[VAL_52]]#0 : index
// CHECK:           %[[VAL_16]] = br %[[VAL_53]] : index
// CHECK:           %[[VAL_22]] = br %[[VAL_54]] : none
// CHECK:           %[[VAL_24]] = br %[[VAL_56]] : index
// CHECK:           %[[VAL_57:.*]], %[[VAL_58:.*]] = control_merge %[[VAL_33]] : none
// CHECK:           sink %[[VAL_58]] : index
// CHECK:           return %[[VAL_57]] : none
// CHECK:         }
func @load_store () -> () {
  %A = memref.alloc() : memref<10xf32>
  affine.for %i = 0 to 10 {
    %0 = affine.load %A[%i] : memref<10xf32>
    affine.store %0, %A[%i] : memref<10xf32>
  }
  return
}


// TODO: affine expr in loop bounds
// TODO: nested loops
// TODO: yield carries values
