// RUN: circt-opt -create-dataflow %s | FileCheck %s
func @bar(%0 : i32) -> i32 {
// CHECK-LABEL:   handshake.func @bar(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none) {
// CHECK:           %[[VAL_2:.*]] = "handshake.merge"(%[[VAL_0]]) : (i32) -> i32
// CHECK:           handshake.return %[[VAL_2]], %[[VAL_1]] : i32, none
// CHECK:         }

  return %0 : i32
}

func @foo(%0 : i32) -> i32 {
// CHECK-LABEL:   handshake.func @foo(
// CHECK-SAME:                        %[[VAL_0:.*]]: i32,
// CHECK-SAME:                        %[[VAL_1:.*]]: none, ...) -> (i32, none) {
// CHECK:           %[[VAL_2:.*]] = "handshake.merge"(%[[VAL_0]]) : (i32) -> i32
// CHECK:           %[[VAL_3:.*]] = handshake.instance @bar(%[[VAL_2]]) : (i32) -> i32
// CHECK:           handshake.return %[[VAL_3]], %[[VAL_1]] : i32, none
// CHECK:         }

  %a1 = call @bar(%0) : (i32) -> i32
  return %a1 : i32
}
