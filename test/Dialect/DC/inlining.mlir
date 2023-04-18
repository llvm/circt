// RUN: circt-opt %s --inline | FileCheck %s

dc.func @foo(%arg0: !dc.value<i64>, %arg1: !dc.value<i64>) -> (!dc.value<i64>) {
  %token, %outputs = dc.unpack %arg0 : (!dc.value<i64>) -> i64
  %token_0, %outputs_1 = dc.unpack %arg1 : (!dc.value<i64>) -> i64
  %0 = arith.cmpi slt, %outputs, %outputs_1 : i64
  %1 = dc.join %token, %token_0
  %2 = arith.select %0, %outputs_1, %outputs : i64
  %3 = dc.pack %1[%2] : (i64) -> !dc.value<i64>
  dc.return %3 : !dc.value<i64>
}

// CHECK-LABEL:   dc.func @bar(
// CHECK-SAME:                 %[[VAL_0:.*]]: !dc.value<i64>,
// CHECK-SAME:                 %[[VAL_1:.*]]: !dc.value<i64>) -> (!dc.value<i64>, !dc.value<i64>) {
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = dc.unpack %[[VAL_0]] : (!dc.value<i64>) -> i64
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = dc.unpack %[[VAL_1]] : (!dc.value<i64>) -> i64
// CHECK:           %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_3]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_7:.*]] = dc.join %[[VAL_2]], %[[VAL_4]]
// CHECK:           %[[VAL_8:.*]] = arith.select %[[VAL_6]], %[[VAL_5]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_9:.*]] = dc.pack %[[VAL_7]]{{\[}}%[[VAL_8]]] : (i64) -> !dc.value<i64>
// CHECK:           dc.return %[[VAL_9]], %[[VAL_9]] : !dc.value<i64>, !dc.value<i64>
// CHECK:         }
dc.func @bar(%a : !dc.value<i64>, %b : !dc.value<i64>) -> (!dc.value<i64>, !dc.value<i64>) {
    %3 = dc.call @foo(%a, %b) : (!dc.value<i64>, !dc.value<i64>) -> (!dc.value<i64>)
    dc.return %3, %3 : !dc.value<i64>, !dc.value<i64>
}

