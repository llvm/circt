// RUN: circt-opt -split-input-file --handshake-materialize-forks-sinks %s | FileCheck %s

// CHECK-LABEL:   handshake.func @missing_fork_and_sink(
// CHECK-SAME:                                          %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                          %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["arg0", "ctrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]]:2 = fork [2] %[[VAL_0]] : i32
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_2]]#0, %[[VAL_2]]#1 : i32
// CHECK:           sink %[[VAL_3]] : i32
// CHECK:           return %[[VAL_1]] : none
// CHECK:         }
handshake.func @missing_fork_and_sink(%arg0 : i32, %ctrl: none) -> (none) {
  %0 = arith.addi %arg0, %arg0 : i32
  return %ctrl: none
}
