// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s

// CHECK:   handshake.func @multi_cond(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index, %[[VAL_4:.*]]: none, ...) -> none attributes {argNames = ["in0", "in1", "in2", "in3", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = merge %[[VAL_1]] : index
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_2]] : index
// CHECK:           %[[VAL_9:.*]] = merge %[[VAL_3]] : index
// CHECK:           %[[VAL_10:.*]]:8 = fork [8] %[[VAL_4]] : none
// CHECK:           %[[VAL_11:.*]] = constant %[[VAL_10]]#6 {value = 0 : index} : index
// CHECK:           %[[VAL_12:.*]]:6 = fork [6] %[[VAL_11]] : index
// CHECK:           %[[VAL_13:.*]] = constant %[[VAL_10]]#5 {value = -1 : index} : index
// CHECK:           %[[VAL_14:.*]] = arith.muli %[[VAL_12]]#0, %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_6]]#1 : index
// CHECK:           %[[VAL_16:.*]] = constant %[[VAL_10]]#4 {value = 1 : index} : index
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : index
// CHECK:           %[[VAL_18:.*]] = arith.cmpi sge, %[[VAL_17]], %[[VAL_12]]#1 : index
// CHECK:           %[[VAL_19:.*]] = constant %[[VAL_10]]#3 {value = -1 : index} : index
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_6]]#0, %[[VAL_19]] : index
// CHECK:           %[[VAL_21:.*]] = arith.cmpi sge, %[[VAL_20]], %[[VAL_12]]#2 : index
// CHECK:           %[[VAL_22:.*]] = arith.andi %[[VAL_18]], %[[VAL_21]] : i1
// CHECK:           %[[VAL_23:.*]] = constant %[[VAL_10]]#2 {value = -1 : index} : index
// CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_7]], %[[VAL_23]] : index
// CHECK:           %[[VAL_25:.*]] = arith.cmpi sge, %[[VAL_24]], %[[VAL_12]]#3 : index
// CHECK:           %[[VAL_26:.*]] = arith.andi %[[VAL_22]], %[[VAL_25]] : i1
// CHECK:           %[[VAL_27:.*]] = constant %[[VAL_10]]#1 {value = -1 : index} : index
// CHECK:           %[[VAL_28:.*]] = arith.addi %[[VAL_8]], %[[VAL_27]] : index
// CHECK:           %[[VAL_29:.*]] = arith.cmpi sge, %[[VAL_28]], %[[VAL_12]]#4 : index
// CHECK:           %[[VAL_30:.*]] = arith.andi %[[VAL_26]], %[[VAL_29]] : i1
// CHECK:           %[[VAL_31:.*]] = constant %[[VAL_10]]#0 {value = -42 : index} : index
// CHECK:           %[[VAL_32:.*]] = arith.addi %[[VAL_9]], %[[VAL_31]] : index
// CHECK:           %[[VAL_33:.*]] = arith.cmpi eq, %[[VAL_32]], %[[VAL_12]]#5 : index
// CHECK:           %[[VAL_34:.*]] = arith.andi %[[VAL_30]], %[[VAL_33]] : i1
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = cond_br %[[VAL_34]], %[[VAL_10]]#7 : none
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]] = control_merge %[[VAL_35]] : none
// CHECK:           sink %[[VAL_38]] : index
// CHECK:           %[[VAL_39:.*]] = br %[[VAL_37]] : none
// CHECK:           %[[VAL_40:.*]], %[[VAL_41:.*]] = control_merge %[[VAL_36]] : none
// CHECK:           sink %[[VAL_41]] : index
// CHECK:           %[[VAL_42:.*]] = br %[[VAL_40]] : none
// CHECK:           %[[VAL_43:.*]], %[[VAL_44:.*]] = control_merge %[[VAL_42]], %[[VAL_39]] : none
// CHECK:           sink %[[VAL_44]] : index
// CHECK:           return %[[VAL_43]] : none
// CHECK:         }
func @multi_cond(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
    %c0 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %1 = arith.muli %c0, %c-1 : index
    %2 = arith.addi %1, %arg0 : index
    %c1 = arith.constant 1 : index
    %3 = arith.addi %2, %c1 : index
    %4 = arith.cmpi sge, %3, %c0 : index
    %c-1_0 = arith.constant -1 : index
    %5 = arith.addi %arg0, %c-1_0 : index
    %6 = arith.cmpi sge, %5, %c0 : index
    %7 = arith.andi %4, %6 : i1
    %c-1_1 = arith.constant -1 : index
    %8 = arith.addi %arg1, %c-1_1 : index
    %9 = arith.cmpi sge, %8, %c0 : index
    %10 = arith.andi %7, %9 : i1
    %c-1_2 = arith.constant -1 : index
    %11 = arith.addi %arg2, %c-1_2 : index
    %12 = arith.cmpi sge, %11, %c0 : index
    %13 = arith.andi %10, %12 : i1
    %c-42 = arith.constant -42 : index
    %14 = arith.addi %arg3, %c-42 : index
    %15 = arith.cmpi eq, %14, %c0 : index
    %16 = arith.andi %13, %15 : i1
    cf.cond_br %16, ^bb1, ^bb2
  ^bb1: // pred: ^bb0
    cf.br ^bb3
  ^bb2: // pred: ^bb0
    cf.br ^bb3
  ^bb3: // 2 preds: ^bb1, ^bb2
    return
  }
