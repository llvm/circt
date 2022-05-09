// RUN: circt-opt -split-input-file %s | circt-opt | FileCheck %s

// CHECK-LABEL:   handshake.func private @private_func(
// CHECK-SAME:                   %[[VAL_0:.*]]: i32,
// CHECK-SAME:                   %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["arg0", "ctrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : i32, none
// CHECK:         }
handshake.func private @private_func(%arg0 : i32, %ctrl: none) -> (i32, none) {
  return %arg0, %ctrl : i32, none
}

// -----

// CHECK-LABEL:   handshake.func public @public_func(
// CHECK-SAME:                   %[[VAL_0:.*]]: i32,
// CHECK-SAME:                   %[[VAL_1:.*]]: none, ...) -> (i32, none) attributes {argNames = ["arg0", "ctrl"], resNames = ["out0", "outCtrl"]} {
// CHECK:           return %[[VAL_0]], %[[VAL_1]] : i32, none
// CHECK:         }
handshake.func public @public_func(%arg0 : i32, %ctrl: none) -> (i32, none) {
  return %arg0, %ctrl : i32, none
}
