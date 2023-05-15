// RUN: circt-opt -pass-pipeline="builtin.module(func.func(dc-materialize-forks-sinks))" %s | FileCheck %s

// CHECK-LABEL:   func.func @testFork(
// CHECK-SAME:                        %[[VAL_0:.*]]: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
// CHECK:           %[[VAL_1:.*]]:3 = dc.fork [3] %[[VAL_0]]
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1, %[[VAL_1]]#2 : !dc.token, !dc.token, !dc.token
// CHECK:         }
func.func @testFork(%a: !dc.token) -> (!dc.token, !dc.token, !dc.token) {
    return %a, %a, %a : !dc.token, !dc.token, !dc.token
}

// CHECK-LABEL:   func.func @testSink(
// CHECK-SAME:                        %[[VAL_0:.*]]: !dc.value<i1>) -> !dc.token {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = dc.branch %[[VAL_0]]
// CHECK:           dc.sink %[[VAL_2]]
// CHECK:           return %[[VAL_1]] : !dc.token
// CHECK:         }
func.func @testSink(%a: !dc.value<i1>) -> (!dc.token) {
    %true, %false = dc.branch %a
    return %true : !dc.token
}
