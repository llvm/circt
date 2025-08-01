// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py

// RUN: circt-opt -lower-cf-to-handshake %s --split-input-file | FileCheck %s

// CHECK-LABEL:   handshake.func @simpleDiamond(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                  %[[VAL_1:.*]]: i64,
// CHECK-SAME:                                  %[[VAL_2:.*]]: none, ...) -> none
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : i1
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] : i64
// CHECK:           %[[VAL_2x:.*]] = merge %[[VAL_2]] : none
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_3]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_3]], %[[VAL_2x]] : none
// CHECK:           %[[VAL_12:.*]] = merge %[[VAL_6]] : i64
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_8]] : none, index
// CHECK:           %[[VAL_14:.*]] = br %[[VAL_12]] : i64
// CHECK:           %[[VAL_13:.*]] = br %[[VAL_10]] : none
// CHECK:           %[[VAL_17:.*]] = merge %[[VAL_7]] : i64
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = control_merge %[[VAL_9]] : none, index
// CHECK:           %[[VAL_19:.*]] = br %[[VAL_17]] : i64
// CHECK:           %[[VAL_18:.*]] = br %[[VAL_15]] : none
// CHECK:           %[[VAL_22:.*]] = mux %[[VAL_21:.*]] {{\[}}%[[VAL_19]], %[[VAL_14]]] : index, i64
// CHECK:           %[[VAL_20:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_18]], %[[VAL_13]]] : i1, none
// CHECK:           %[[VAL_21]] = arith.index_cast %[[VAL_4]] : i1 to index
// CHECK:           return %[[VAL_20]] : none
// CHECK:         }
func.func @simpleDiamond(%arg0: i1, %arg1: i64) {
  cf.cond_br %arg0, ^bb1(%arg1: i64), ^bb2(%arg1: i64)
^bb1(%v1: i64):  // pred: ^bb0
  cf.br ^bb3(%v1: i64)
^bb2(%v2: i64):  // pred: ^bb0
  cf.br ^bb3(%v2: i64)
^bb3(%v3: i64):  // 2 preds: ^bb1, ^bb2
  return
}

// -----

// CHECK-LABEL:   handshake.func @nestedDiamond(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                  %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = buffer [2] fifo %[[VAL_2]] : i1
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_2]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_2]], %[[VAL_1x]] : none
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_4]] : i1
// CHECK:           %[[VAL_9:.*]] = buffer [2] fifo %[[VAL_8]] : i1
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_6]] : none, index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_8]], %[[VAL_10]] : none
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = control_merge %[[VAL_12]] : none, index
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_14]] : none
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = control_merge %[[VAL_13]] : none, index
// CHECK:           %[[VAL_19:.*]] = br %[[VAL_17]] : none
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = control_merge %[[VAL_7]] : none, index
// CHECK:           %[[VAL_22:.*]] = br %[[VAL_20]] : none
// CHECK:           %[[VAL_23:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_19]], %[[VAL_16]]] : i1, none
// CHECK:           %[[VAL_24:.*]] = arith.index_cast %[[VAL_9]] : i1 to index
// CHECK:           %[[VAL_25:.*]] = br %[[VAL_23]] : none
// CHECK:           %[[VAL_26:.*]] = mux %[[VAL_3]] {{\[}}%[[VAL_22]], %[[VAL_25]]] : i1, none
// CHECK:           %[[VAL_29:.*]] = arith.index_cast %[[VAL_3]] : i1 to index
// CHECK:           return %[[VAL_26]] : none
// CHECK:         }
func.func @nestedDiamond(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb5
^bb3:  // pred: ^bb1
  cf.br ^bb5
^bb4:  // pred: ^bb0
  cf.br ^bb6
^bb5:  // 2 preds: ^bb2, ^bb3
  cf.br ^bb6
^bb6:  // 2 preds: ^bb4, ^bb5
  return
}

// -----

// CHECK-LABEL:   handshake.func @triangle(
// CHECK-SAME:                             %[[VAL_0:.*]]: i1,
// CHECK-SAME:                             %[[VAL_1:.*]]: i64,
// CHECK-SAME:                             %[[VAL_2:.*]]: none, ...) -> none
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : i1
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] : i64
// CHECK:           %[[VAL_2x:.*]] = merge %[[VAL_2]] : none
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_3]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_3]], %[[VAL_2x]] : none
// CHECK:           %[[VAL_12:.*]] = merge %[[VAL_6]] : i64
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_8]] : none, index
// CHECK:           %[[VAL_14:.*]] = br %[[VAL_12]] : i64
// CHECK:           %[[VAL_13:.*]] = br %[[VAL_10]] : none
// CHECK:           %[[VAL_19:.*]] = mux %[[VAL_18:.*]] {{\[}}%[[VAL_14]], %[[VAL_7]]] : index, i64
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_9]], %[[VAL_13]]] : i1, none
// CHECK:           %[[VAL_16:.*]] = constant %[[VAL_15]] {value = true} : i1
// CHECK:           %[[VAL_17:.*]] = arith.xori %[[VAL_4]], %[[VAL_16]] : i1
// CHECK:           %[[VAL_18]] = arith.index_cast %[[VAL_17]] : i1 to index
// CHECK:           return %[[VAL_15]] : none
// CHECK:         }
func.func @triangle(%arg0: i1, %val0: i64) {
  cf.cond_br %arg0, ^bb1(%val0: i64), ^bb2(%val0: i64)
^bb1(%val1: i64):  // pred: ^bb0
  cf.br ^bb2(%val1: i64)
^bb2(%val2: i64):  // 2 preds: ^bb0, ^bb1
  return
}

// -----

// CHECK-LABEL:   handshake.func @nestedTriangle(
// CHECK-SAME:                                   %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                   %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = buffer [2] fifo %[[VAL_2]] : i1
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_2]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_2]], %[[VAL_1x]] : none
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_4]] : i1
// CHECK:           %[[VAL_9:.*]] = buffer [2] fifo %[[VAL_8]] : i1
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_6]] : none, index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_8]], %[[VAL_10]] : none
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = control_merge %[[VAL_12]] : none, index
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_14]] : none
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_13]], %[[VAL_16]]] : i1, none
// CHECK:           %[[VAL_18:.*]] = constant %[[VAL_17]] {value = true} : i1
// CHECK:           %[[VAL_19:.*]] = arith.xori %[[VAL_9]], %[[VAL_18]] : i1
// CHECK:           %[[VAL_20:.*]] = arith.index_cast %[[VAL_19]] : i1 to index
// CHECK:           %[[VAL_21:.*]] = br %[[VAL_17]] : none
// CHECK:           %[[VAL_22:.*]] = mux %[[VAL_3]] {{\[}}%[[VAL_7]], %[[VAL_21]]] : i1, none
// CHECK:           %[[VAL_23:.*]] = constant %[[VAL_22]] {value = true} : i1
// CHECK:           %[[VAL_24:.*]] = arith.xori %[[VAL_3]], %[[VAL_23]] : i1
// CHECK:           %[[VAL_25:.*]] = arith.index_cast %[[VAL_24]] : i1 to index
// CHECK:           return %[[VAL_22]] : none
// CHECK:         }
func.func @nestedTriangle(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // 2 preds: ^bb0, ^bb3
  return
}

// -----

// CHECK-LABEL:   handshake.func @multiple_blocks_needed(
// CHECK-SAME:                                           %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                           %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = buffer [2] fifo %[[VAL_2]] : i1
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_2]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_2]], %[[VAL_1x]] : none
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_4]] : i1
// CHECK:           %[[VAL_9:.*]] = buffer [2] fifo %[[VAL_8]] : i1
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_6]] : none, index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_8]], %[[VAL_8]] : i1
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = cond_br %[[VAL_8]], %[[VAL_10]] : none
// CHECK:           %[[VAL_16:.*]] = merge %[[VAL_12]] : i1
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = control_merge %[[VAL_14]] : none, index
// CHECK:           %[[VAL_19:.*]] = br %[[VAL_16]] : i1
// CHECK:           %[[VAL_20:.*]] = br %[[VAL_17]] : none
// CHECK:           %[[VAL_21:.*]] = mux %[[VAL_22:.*]] {{\[}}%[[VAL_19]], %[[VAL_13]]] : index, i1
// CHECK:           %[[VAL_23:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_15]], %[[VAL_20]]] : i1, none
// CHECK:           %[[VAL_24:.*]] = constant %[[VAL_23]] {value = true} : i1
// CHECK:           %[[VAL_25:.*]] = arith.xori %[[VAL_9]], %[[VAL_24]] : i1
// CHECK:           %[[VAL_22]] = arith.index_cast %[[VAL_25]] : i1 to index
// CHECK:           %[[VAL_26:.*]] = br %[[VAL_21]] : i1
// CHECK:           %[[VAL_27:.*]] = br %[[VAL_23]] : none
// CHECK:           %[[VAL_28:.*]] = mux %[[VAL_29:.*]] {{\[}}%[[VAL_26]], %[[VAL_5]]] : index, i1
// CHECK:           %[[VAL_30:.*]] = buffer [2] fifo %[[VAL_28]] : i1
// CHECK:           %[[VAL_31:.*]] = mux %[[VAL_3]] {{\[}}%[[VAL_7]], %[[VAL_27]]] : i1, none
// CHECK:           %[[VAL_32:.*]] = constant %[[VAL_31]] {value = true} : i1
// CHECK:           %[[VAL_33:.*]] = arith.xori %[[VAL_3]], %[[VAL_32]] : i1
// CHECK:           %[[VAL_29]] = arith.index_cast %[[VAL_33]] : i1 to index
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = cond_br %[[VAL_28]], %[[VAL_28]] : i1
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = cond_br %[[VAL_28]], %[[VAL_31]] : none
// CHECK:           %[[VAL_38:.*]] = merge %[[VAL_34]] : i1
// CHECK:           %[[VAL_39:.*]] = buffer [2] fifo %[[VAL_38]] : i1
// CHECK:           %[[VAL_40:.*]], %[[VAL_41:.*]] = control_merge %[[VAL_36]] : none, index
// CHECK:           %[[VAL_42:.*]], %[[VAL_43:.*]] = cond_br %[[VAL_38]], %[[VAL_40]] : none
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = control_merge %[[VAL_42]] : none, index
// CHECK:           %[[VAL_46:.*]] = br %[[VAL_44]] : none
// CHECK:           %[[VAL_47:.*]] = mux %[[VAL_39]] {{\[}}%[[VAL_43]], %[[VAL_46]]] : i1, none
// CHECK:           %[[VAL_48:.*]] = constant %[[VAL_47]] {value = true} : i1
// CHECK:           %[[VAL_49:.*]] = arith.xori %[[VAL_39]], %[[VAL_48]] : i1
// CHECK:           %[[VAL_50:.*]] = arith.index_cast %[[VAL_49]] : i1 to index
// CHECK:           %[[VAL_51:.*]] = br %[[VAL_47]] : none
// CHECK:           %[[VAL_52:.*]] = mux %[[VAL_30]] {{\[}}%[[VAL_37]], %[[VAL_51]]] : i1, none
// CHECK:           %[[VAL_53:.*]] = constant %[[VAL_52]] {value = true} : i1
// CHECK:           %[[VAL_54:.*]] = arith.xori %[[VAL_30]], %[[VAL_53]] : i1
// CHECK:           %[[VAL_55:.*]] = arith.index_cast %[[VAL_54]] : i1 to index
// CHECK:           return %[[VAL_52]] : none
// CHECK:         }
func.func @multiple_blocks_needed(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  cf.br ^bb4
^bb4:  // 2 preds: ^bb0, ^bb3
  cf.cond_br %arg0, ^bb5, ^bb8
^bb5:  // pred: ^bb4
  cf.cond_br %arg0, ^bb6, ^bb7
^bb6:  // pred: ^bb5
  cf.br ^bb7
^bb7:  // 2 preds: ^bb5, ^bb6
  cf.br ^bb8
^bb8:  // 2 preds: ^bb4, ^bb7
  return
}

// -----

// CHECK-LABEL:   handshake.func @sameSuccessor(
// CHECK-SAME:                                  %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                  %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = cond_br %[[VAL_2]], %[[VAL_1x]] : none
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = cond_br %[[VAL_2]], %[[VAL_1x]] : none
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = control_merge %[[VAL_3]], %[[VAL_3]] : none, index
// CHECK:           return %[[VAL_7]] : none
// CHECK:         }
func.func @sameSuccessor(%cond: i1) {
  cf.cond_br %cond, ^1, ^1
^1:
  return
}

// -----

// CHECK-LABEL:   handshake.func @simple_loop(
// CHECK-SAME:                                %[[VAL_0:.*]]: i64,
// CHECK-SAME:                                %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i64
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_1x]] {value = 1 : i64} : i64
// CHECK:           %[[VAL_4:.*]] = br %[[VAL_2]] : i64
// CHECK:           %[[VAL_5:.*]] = br %[[VAL_1x]] : none
// CHECK:           %[[VAL_6:.*]] = br %[[VAL_3]] : i64
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = control_merge %[[VAL_5]] : none, index
// CHECK:           %[[VAL_9:.*]] = buffer [1] seq %[[VAL_10:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_11:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_7]], %[[VAL_12:.*]]] : i1, none
// CHECK:           %[[VAL_16:.*]] = mux %[[VAL_8]] {{\[}}%[[VAL_6]]] : index, i64
// CHECK:           %[[VAL_17:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_16]], %[[VAL_18:.*]]] : i1, i64
// CHECK:           %[[VAL_13:.*]] = mux %[[VAL_8]] {{\[}}%[[VAL_4]]] : index, i64
// CHECK:           %[[VAL_14:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_13]], %[[VAL_15:.*]]] : i1, i64
// CHECK:           %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_17]], %[[VAL_14]] : i64
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = cond_br %[[VAL_19]], %[[VAL_17]] : i64
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = cond_br %[[VAL_19]], %[[VAL_14]] : i64
// CHECK:           %[[VAL_22:.*]] = constant %[[VAL_11]] {value = true} : i1
// CHECK:           %[[VAL_23:.*]] = arith.xori %[[VAL_19]], %[[VAL_22]] : i1
// CHECK:           %[[VAL_10]] = merge %[[VAL_23]] : i1
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]] = cond_br %[[VAL_19]], %[[VAL_11]] : none
// CHECK:           %[[VAL_29:.*]] = merge %[[VAL_21]] : i64
// CHECK:           %[[VAL_28:.*]] = merge %[[VAL_27]] : i64
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = control_merge %[[VAL_25]] : none, index
// CHECK:           %[[VAL_32:.*]] = constant %[[VAL_30]] {value = 1 : i64} : i64
// CHECK:           %[[VAL_33:.*]] = arith.addi %[[VAL_28]], %[[VAL_32]] : i64
// CHECK:           %[[VAL_15]] = br %[[VAL_29]] : i64
// CHECK:           %[[VAL_12]] = br %[[VAL_30]] : none
// CHECK:           %[[VAL_18]] = br %[[VAL_33]] : i64
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = control_merge %[[VAL_24]] : none, index
// CHECK:           return %[[VAL_34]] : none
// CHECK:         }
func.func @simple_loop(%arg0: i64) {
  %c1_i64 = arith.constant 1 : i64
  cf.br ^bb1(%c1_i64 : i64)
^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
  %1 = arith.cmpi eq, %0, %arg0 : i64
  cf.cond_br %1, ^bb3, ^bb2
^bb2:  // pred: ^bb1
  %c1_i64_0 = arith.constant 1 : i64
  %2 = arith.addi %0, %c1_i64_0 : i64
  cf.br ^bb1(%2 : i64)
^bb3:  // pred: ^bb1
  return
}

// -----

// CHECK-LABEL:   handshake.func @blockWith3PredsAndLoop(
// CHECK-SAME:                                           %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                           %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = buffer [2] fifo %[[VAL_2]] : i1
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_2]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_2]], %[[VAL_1x]] : none
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_4]] : i1
// CHECK:           %[[VAL_9:.*]] = buffer [2] fifo %[[VAL_8]] : i1
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_6]] : none, index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_8]], %[[VAL_10]] : none
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = control_merge %[[VAL_12]] : none, index
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_14]] : none
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = control_merge %[[VAL_13]] : none, index
// CHECK:           %[[VAL_19:.*]] = br %[[VAL_17]] : none
// CHECK:           %[[VAL_20:.*]] = merge %[[VAL_5]] : i1
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]] = control_merge %[[VAL_7]] : none, index
// CHECK:           %[[VAL_23:.*]] = br %[[VAL_20]] : i1
// CHECK:           %[[VAL_24:.*]] = br %[[VAL_21]] : none
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = control_merge %[[VAL_24]] : none, index
// CHECK:           %[[VAL_27:.*]] = buffer [1] seq %[[VAL_28:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_29:.*]] = mux %[[VAL_27]] {{\[}}%[[VAL_25]], %[[VAL_30:.*]]] : i1, none
// CHECK:           %[[VAL_31:.*]] = mux %[[VAL_26]] {{\[}}%[[VAL_23]]] : index, i1
// CHECK:           %[[VAL_32:.*]] = mux %[[VAL_27]] {{\[}}%[[VAL_31]], %[[VAL_33:.*]]] : i1, i1
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = cond_br %[[VAL_32]], %[[VAL_32]] : i1
// CHECK:           %[[VAL_36:.*]] = constant %[[VAL_29]] {value = true} : i1
// CHECK:           %[[VAL_37:.*]] = arith.xori %[[VAL_32]], %[[VAL_36]] : i1
// CHECK:           %[[VAL_28]] = merge %[[VAL_37]] : i1
// CHECK:           %[[VAL_38:.*]], %[[VAL_39:.*]] = cond_br %[[VAL_32]], %[[VAL_29]] : none
// CHECK:           %[[VAL_40:.*]] = merge %[[VAL_35]] : i1
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = control_merge %[[VAL_39]] : none, index
// CHECK:           %[[VAL_33]] = br %[[VAL_40]] : i1
// CHECK:           %[[VAL_30]] = br %[[VAL_41]] : none
// CHECK:           %[[VAL_43:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_19]], %[[VAL_16]]] : i1, none
// CHECK:           %[[VAL_44:.*]] = arith.index_cast %[[VAL_9]] : i1 to index
// CHECK:           %[[VAL_45:.*]] = br %[[VAL_43]] : none
// CHECK:           %[[VAL_46:.*]] = mux %[[VAL_3]] {{\[}}%[[VAL_38]], %[[VAL_45]]] : i1, none
// CHECK:           %[[VAL_49:.*]] = arith.index_cast %[[VAL_3]] : i1 to index
// CHECK:           return %[[VAL_46]] : none
// CHECK:         }
func.func @blockWith3PredsAndLoop(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb7
^bb3:  // pred: ^bb1
  cf.br ^bb7
^bb4:  // pred: ^bb0
  cf.br ^bb5
^bb5:  // 2 preds: ^bb4, ^bb6
  cf.cond_br %arg0, ^bb8, ^bb6
^bb6:  // pred: ^bb5
  cf.br ^bb5
^bb7:  // 2 preds: ^bb2, ^bb3
  cf.br ^bb8
^bb8:  // 2 preds: ^bb5, ^bb7
  return
}

// -----

// CHECK-LABEL:   handshake.func @otherBlockOrder(
// CHECK-SAME:                                    %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                    %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = buffer [2] fifo %[[VAL_2]] : i1
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_2]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_2]], %[[VAL_1x]] : none
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_4]] : i1
// CHECK:           %[[VAL_9:.*]] = buffer [2] fifo %[[VAL_8]] : i1
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_6]] : none, index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_8]], %[[VAL_10]] : none
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = control_merge %[[VAL_12]] : none, index
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_14]] : none
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = control_merge %[[VAL_13]] : none, index
// CHECK:           %[[VAL_19:.*]] = br %[[VAL_17]] : none
// CHECK:           %[[VAL_20:.*]] = merge %[[VAL_5]] : i1
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]] = control_merge %[[VAL_7]] : none, index
// CHECK:           %[[VAL_23:.*]] = br %[[VAL_20]] : i1
// CHECK:           %[[VAL_24:.*]] = br %[[VAL_21]] : none
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = control_merge %[[VAL_24]] : none, index
// CHECK:           %[[VAL_27:.*]] = buffer [1] seq %[[VAL_28:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_29:.*]] = mux %[[VAL_27]] {{\[}}%[[VAL_25]], %[[VAL_30:.*]]] : i1, none
// CHECK:           %[[VAL_31:.*]] = mux %[[VAL_26]] {{\[}}%[[VAL_23]]] : index, i1
// CHECK:           %[[VAL_32:.*]] = mux %[[VAL_27]] {{\[}}%[[VAL_31]], %[[VAL_33:.*]]] : i1, i1
// CHECK:           %[[VAL_34:.*]] = br %[[VAL_32]] : i1
// CHECK:           %[[VAL_35:.*]] = br %[[VAL_29]] : none
// CHECK:           %[[VAL_36:.*]] = merge %[[VAL_34]] : i1
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]] = control_merge %[[VAL_35]] : none, index
// CHECK:           %[[VAL_39:.*]], %[[VAL_33]] = cond_br %[[VAL_36]], %[[VAL_36]] : i1
// CHECK:           %[[VAL_40:.*]] = constant %[[VAL_37]] {value = true} : i1
// CHECK:           %[[VAL_41:.*]] = arith.xori %[[VAL_36]], %[[VAL_40]] : i1
// CHECK:           %[[VAL_28]] = merge %[[VAL_41]] : i1
// CHECK:           %[[VAL_42:.*]], %[[VAL_30]] = cond_br %[[VAL_36]], %[[VAL_37]] : none
// CHECK:           %[[VAL_43:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_19]], %[[VAL_16]]] : i1, none
// CHECK:           %[[VAL_44:.*]] = arith.index_cast %[[VAL_9]] : i1 to index
// CHECK:           %[[VAL_45:.*]] = br %[[VAL_43]] : none
// CHECK:           %[[VAL_46:.*]] = mux %[[VAL_3]] {{\[}}%[[VAL_42]], %[[VAL_45]]] : i1, none
// CHECK:           %[[VAL_49:.*]] = arith.index_cast %[[VAL_3]] : i1 to index
// CHECK:           return %[[VAL_46]] : none
// CHECK:         }
func.func @otherBlockOrder(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:  // pred: ^bb0
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  cf.br ^bb7
^bb3:  // pred: ^bb1
  cf.br ^bb7
^bb4:  // pred: ^bb0
  cf.br ^bb5
^bb5:  // 2 preds: ^bb4, ^bb6
  cf.br ^bb6
^bb6:  // pred: ^bb5
  cf.cond_br %arg0, ^bb8, ^bb5
^bb7:  // 2 preds: ^bb2, ^bb3
  cf.br ^bb8
^bb8:  // 2 preds: ^bb6, ^bb7
  return
}

// -----

// CHECK-LABEL:   handshake.func @multiple_block_args(
// CHECK-SAME:                                        %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                        %[[VAL_1:.*]]: i64,
// CHECK-SAME:                                        %[[VAL_2:.*]]: none, ...) -> none
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_4:.*]] = buffer [2] fifo %[[VAL_3]] : i1
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] : i64
// CHECK:           %[[VAL_2x:.*]] = merge %[[VAL_2]] : none
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_3]], %[[VAL_3]] : i1
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = cond_br %[[VAL_3]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = cond_br %[[VAL_3]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = cond_br %[[VAL_3]], %[[VAL_2x]] : none
// CHECK:           %[[VAL_18:.*]] = merge %[[VAL_8]] : i64
// CHECK:           %[[VAL_14:.*]] = merge %[[VAL_6]] : i1
// CHECK:           %[[VAL_15:.*]] = buffer [2] fifo %[[VAL_14]] : i1
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = control_merge %[[VAL_12]] : none, index
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]] = cond_br %[[VAL_14]], %[[VAL_18]] : i64
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = cond_br %[[VAL_14]], %[[VAL_18]] : i64
// CHECK:           %[[VAL_19:.*]], %[[VAL_20:.*]] = cond_br %[[VAL_14]], %[[VAL_16]] : none
// CHECK:           %[[VAL_27:.*]] = merge %[[VAL_21]] : i64
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = control_merge %[[VAL_19]] : none, index
// CHECK:           %[[VAL_28:.*]] = br %[[VAL_25]] : none
// CHECK:           %[[VAL_31:.*]] = merge %[[VAL_24]] : i64
// CHECK:           %[[VAL_32:.*]] = merge %[[VAL_22]] : i64
// CHECK:           %[[VAL_29:.*]], %[[VAL_30:.*]] = control_merge %[[VAL_20]] : none, index
// CHECK:           %[[VAL_33:.*]] = br %[[VAL_29]] : none
// CHECK:           %[[VAL_36:.*]] = merge %[[VAL_11]] : i64
// CHECK:           %[[VAL_37:.*]] = merge %[[VAL_9]] : i64
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = control_merge %[[VAL_13]] : none, index
// CHECK:           %[[VAL_38:.*]] = br %[[VAL_34]] : none
// CHECK:           %[[VAL_39:.*]] = mux %[[VAL_15]] {{\[}}%[[VAL_33]], %[[VAL_28]]] : i1, none
// CHECK:           %[[VAL_40:.*]] = arith.index_cast %[[VAL_15]] : i1 to index
// CHECK:           %[[VAL_41:.*]] = br %[[VAL_39]] : none
// CHECK:           %[[VAL_42:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_38]], %[[VAL_41]]] : i1, none
// CHECK:           %[[VAL_45:.*]] = arith.index_cast %[[VAL_4]] : i1 to index
// CHECK:           return %[[VAL_42]] : none
// CHECK:         }
func.func @multiple_block_args(%arg0: i1, %arg1: i64) {
  cf.cond_br %arg0, ^bb1(%arg1 : i64), ^bb4(%arg1, %arg1 : i64, i64)
^bb1(%0: i64):  // pred: ^bb0
  cf.cond_br %arg0, ^bb2(%0 : i64), ^bb3(%0, %0 : i64, i64)
^bb2(%1: i64):  // pred: ^bb1
  cf.br ^bb5
^bb3(%2: i64, %3: i64):  // pred: ^bb1
  cf.br ^bb5
^bb4(%4: i64, %5: i64):  // pred: ^bb0
  cf.br ^bb6
^bb5:  // 2 preds: ^bb2, ^bb3
  cf.br ^bb6
^bb6:  // 2 preds: ^bb4, ^bb5
  return
}

// -----

// CHECK-LABEL:   handshake.func @mergeBlockAsLoopHeader(
// CHECK-SAME:                                           %[[VAL_0:.*]]: i1,
// CHECK-SAME:                                           %[[VAL_1:.*]]: none, ...) -> none
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_3:.*]] = buffer [2] fifo %[[VAL_2]] : i1
// CHECK:           %[[VAL_1x:.*]] = merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = cond_br %[[VAL_2]], %[[VAL_2]] : i1
// CHECK:           %[[VAL_6:.*]], %[[VAL_7:.*]] = cond_br %[[VAL_2]], %[[VAL_1x]] : none
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_4]] : i1
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = control_merge %[[VAL_6]] : none, index
// CHECK:           %[[VAL_11:.*]] = br %[[VAL_8]] : i1
// CHECK:           %[[VAL_12:.*]] = br %[[VAL_9]] : none
// CHECK:           %[[VAL_13:.*]] = merge %[[VAL_5]] : i1
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = control_merge %[[VAL_7]] : none, index
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_13]] : i1
// CHECK:           %[[VAL_17:.*]] = br %[[VAL_14]] : none
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_3]] {{\[}}%[[VAL_17]], %[[VAL_12]]] : i1, none
// CHECK:           %[[VAL_19:.*]] = arith.index_cast %[[VAL_3]] : i1 to index
// CHECK:           %[[VAL_20:.*]] = buffer [1] seq %[[VAL_21:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_22:.*]] = mux %[[VAL_20]] {{\[}}%[[VAL_18]], %[[VAL_23:.*]]] : i1, none
// CHECK:           %[[VAL_24:.*]] = mux %[[VAL_19]] {{\[}}%[[VAL_16]], %[[VAL_11]]] : index, i1
// CHECK:           %[[VAL_25:.*]] = mux %[[VAL_20]] {{\[}}%[[VAL_24]], %[[VAL_26:.*]]] : i1, i1
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]] = cond_br %[[VAL_25]], %[[VAL_25]] : i1
// CHECK:           %[[VAL_29:.*]] = constant %[[VAL_18]] {value = true} : i1
// CHECK:           %[[VAL_30:.*]] = arith.xori %[[VAL_25]], %[[VAL_29]] : i1
// CHECK:           %[[VAL_21]] = merge %[[VAL_30]] : i1
// CHECK:           %[[VAL_31:.*]], %[[VAL_32:.*]] = cond_br %[[VAL_25]], %[[VAL_22]] : none
// CHECK:           %[[VAL_33:.*]] = merge %[[VAL_28]] : i1
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = control_merge %[[VAL_32]] : none, index
// CHECK:           %[[VAL_26]] = br %[[VAL_33]] : i1
// CHECK:           %[[VAL_23]] = br %[[VAL_34]] : none
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = control_merge %[[VAL_31]] : none, index
// CHECK:           return %[[VAL_36]] : none
// CHECK:         }
func.func @mergeBlockAsLoopHeader(%arg0: i1) {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  cf.br ^bb3
^bb2:  // pred: ^bb0
  cf.br ^bb3
^bb3:  // pred: ^bb1, ^bb2, ^bb4
  cf.cond_br %arg0, ^bb5, ^bb4
^bb4:  // pred: ^bb1, ^bb2
  cf.br ^bb3
^bb5:  // 2 preds: ^bb3
  return
}
