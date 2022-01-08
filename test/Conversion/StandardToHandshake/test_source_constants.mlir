// RUN: circt-opt -lower-std-to-handshake="source-constants" %s --canonicalize | FileCheck %s

// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : i32
// CHECK:           %[[VAL_3:.*]] = source
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]] {value = 1 : i32} : i32
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_2]], %[[VAL_4]] : i32
// CHECK:           return %[[VAL_5]], %[[VAL_1]] : i32, none
// CHECK:         }

func @foo(%arg0 : i32) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.addi %arg0, %c1_i32 : i32
  return %0 : i32
}
