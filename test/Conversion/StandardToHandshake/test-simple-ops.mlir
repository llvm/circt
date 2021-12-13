// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s

// CHECK-LABEL:   handshake.func @main(
// CHECK:                         %[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: none, ...) -> (i32, none)
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_0]] : i1
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_2]] : i32
// CHECK:           %[[VAL_7:.*]] = mux %[[VAL_4]] {{\[}}%[[VAL_6]], %[[VAL_5]]] : i1, i32
// CHECK:           return %[[VAL_7]], %[[VAL_3]] : i32, none
// CHECK:         }
func @main(%c : i1, %a : i32, %b : i32) -> i32 {
  %0 = std.select %c, %a, %b : i32
  return %0 : i32
}
