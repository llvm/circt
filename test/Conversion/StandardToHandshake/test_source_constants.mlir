// RUN: circt-opt -lower-std-to-handshake="source-constants" %s --canonicalize | FileCheck %s

// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = source
// CHECK:           %[[VAL_3:.*]] = constant %[[VAL_2]] {value = 1 : i32} : i32
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_0]], %[[VAL_3]] : i32
// CHECK:           return %[[VAL_4]], %[[VAL_1]] : i32, none
// CHECK:         }

func @foo(%arg0 : i32) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %0 = arith.addi %arg0, %c1_i32 : i32
  return %0 : i32
}
