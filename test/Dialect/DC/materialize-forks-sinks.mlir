// RUN: circt-opt %s --dc-materialize-forks-sinks | FileCheck %s

// CHECK-LABEL:   dc.func @testFork(
// CHECK-SAME:                      %[[VAL_0:.*]]: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
// CHECK:           %[[VAL_1:.*]]:3 = dc.fork [3] %[[VAL_0]]
// CHECK:           dc.return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : !dc.token, !dc.token, !dc.token
// CHECK:         }
dc.func @testFork(%a: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
    dc.return %a, %a, %a : !dc.token, !dc.token, !dc.token
}

// CHECK-LABEL:   dc.func @testSink(
// CHECK-SAME:                      %[[VAL_0:.*]]: !dc.value<i1>) -> !dc.token {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = dc.branch %[[VAL_0]]
// CHECK:           dc.sink %[[VAL_2]]
// CHECK:           dc.return %[[VAL_1]] : !dc.token
// CHECK:         }
dc.func @testSink(%a: !dc.value<i1>) -> (!dc.token) {
    %true, %false = dc.branch %a
    dc.return %true : !dc.token
}
