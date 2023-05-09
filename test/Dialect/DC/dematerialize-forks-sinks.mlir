// RUN: circt-opt %s --dc-dematerialize-forks-sinks | FileCheck %s

// CHECK-LABEL:   dc.func @testFork(
// CHECK-SAME:                      %[[VAL_0:.*]]: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
// CHECK:           dc.return %[[VAL_0]], %[[VAL_0]], %[[VAL_0]] : !dc.token, !dc.token, !dc.token
// CHECK:         }
dc.func @testFork(%arg0: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
  %0:3 = dc.fork [3] %arg0 
  dc.return %0#0, %0#1, %0#2 : !dc.token, !dc.token, !dc.token
}

// CHECK-LABEL:   dc.func @testSink(
// CHECK-SAME:                      %[[VAL_0:.*]]: !dc.value<i1>) -> !dc.token {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = dc.branch %[[VAL_0]]
// CHECK:           dc.return %[[VAL_1]] : !dc.token
// CHECK:         }
dc.func @testSink(%arg0: !dc.value<i1>) -> !dc.token {
  %trueToken, %falseToken = dc.branch %arg0
  dc.sink %falseToken
  dc.return %trueToken : !dc.token
}
