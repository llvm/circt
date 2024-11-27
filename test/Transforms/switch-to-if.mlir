// RUN: circt-opt -split-input-file --switch-to-if %s | FileCheck %s

// CHECK-LABEL:   func.func @example(
// CHECK-SAME:                       %[[VAL_0:.*]]: index) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_1]], %[[VAL_0]] : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.cmpi eq, %[[VAL_2]], %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = scf.if %[[VAL_4]] -> (i32) {
// CHECK:             %[[VAL_6:.*]] = arith.constant 10 : i32
// CHECK:             scf.yield %[[VAL_6]] : i32
// CHECK:           } else {
// CHECK:             %[[VAL_7:.*]] = arith.constant 5 : index
// CHECK:             %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_2]], %[[VAL_7]] : index
// CHECK:             %[[VAL_9:.*]] = scf.if %[[VAL_8]] -> (i32) {
// CHECK:               %[[VAL_10:.*]] = arith.constant 20 : i32
// CHECK:               scf.yield %[[VAL_10]] : i32
// CHECK:             } else {
// CHECK:               %[[VAL_11:.*]] = arith.constant 7 : index
// CHECK:               %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_2]], %[[VAL_11]] : index
// CHECK:               %[[VAL_13:.*]] = scf.if %[[VAL_12]] -> (i32) {
// CHECK:                 %[[VAL_14:.*]] = arith.constant 30 : i32
// CHECK:                 scf.yield %[[VAL_14]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_15:.*]] = arith.constant 50 : i32
// CHECK:                 scf.yield %[[VAL_15]] : i32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_13]] : i32
// CHECK:             }
// CHECK:             scf.yield %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }
module {
  func.func @example(%arg0 : index)  -> i32 {
    %one = arith.constant 1 : index
    %cond = arith.addi %one, %arg0 : index
    %0 = scf.index_switch %cond -> i32
    case 2 {
        %1 = arith.constant 10 : i32
        scf.yield %1 : i32
    }
    case 5 {
        %2 = arith.constant 20 : i32
        scf.yield %2 : i32
    }
    case 7 {
        %3 = arith.constant 30 : i32
        scf.yield %3 : i32
    }
    default {
        %4 = arith.constant 50 : i32
        scf.yield %4 : i32
    }
    return %0 : i32
  }
}
