// Note: canonicalization is also run to remove pack/unpack operations.
// If not done, those would clutter the output.
// RUN: circt-opt %s --lower-handshake-to-dc --canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @test_fork(
// CHECK-SAME:                         %[[VAL_0:.*]]: !dc.token) -> (!dc.token, !dc.token) {
// CHECK:           %[[VAL_1:.*]]:2 = dc.fork %[[VAL_0]] : !dc.token, !dc.token
// CHECK:           return %[[VAL_1]]#0, %[[VAL_1]]#1 : !dc.token, !dc.token
// CHECK:         }
handshake.func @test_fork(%arg0: none) -> (none, none) {
  %0:2 = fork [2] %arg0 : none
  return %0#0, %0#1 : none, none
}

// CHECK-LABEL:   func.func @test_fork_data(
// CHECK-SAME:                              %[[VAL_0:.*]]: !dc.value<i32>) -> !dc.value<i32> {
// CHECK:           %[[VAL_1:.*]], %[[VAL_2:.*]] = dc.unpack %[[VAL_0]] : (!dc.value<i32>) -> i32
// CHECK:           %[[VAL_3:.*]]:2 = dc.fork %[[VAL_1]] : !dc.token, !dc.token
// CHECK:           %[[VAL_4:.*]] = dc.join %[[VAL_3]]#0, %[[VAL_3]]#1
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_2]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_6:.*]] = dc.pack %[[VAL_4]]{{\[}}%[[VAL_5]]] : (i32) -> !dc.value<i32>
// CHECK:           return %[[VAL_6]] : !dc.value<i32>
// CHECK:         }
handshake.func @test_fork_data(%arg0: i32) -> (i32) {
  %0:2 = fork [2] %arg0 : i32
  %1 = arith.addi %0#0, %0#1 : i32
  return %1 : i32
}

// CHECK-LABEL:   func.func @top(
// CHECK-SAME:                   %[[VAL_0:.*]]: !dc.value<i64>,
// CHECK-SAME:                   %[[VAL_1:.*]]: !dc.value<i64>,
// CHECK-SAME:                   %[[VAL_2:.*]]: !dc.token) -> (!dc.value<i64>, !dc.token) {
// CHECK:           %[[VAL_3:.*]], %[[VAL_4:.*]] = dc.unpack %[[VAL_0]] : (!dc.value<i64>) -> i64
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = dc.unpack %[[VAL_1]] : (!dc.value<i64>) -> i64
// CHECK:           %[[VAL_7:.*]] = dc.join %[[VAL_3]], %[[VAL_5]]
// CHECK:           %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_4]], %[[VAL_6]] : i64
// CHECK:           %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_6]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_10:.*]] = dc.pack %[[VAL_7]]{{\[}}%[[VAL_9]]] : (i64) -> !dc.value<i64>
// CHECK:           return %[[VAL_10]], %[[VAL_2]] : !dc.value<i64>, !dc.token
// CHECK:         }
handshake.func @top(%arg0: i64, %arg1: i64, %arg8: none, ...) -> (i64, none) {
    %0 = arith.cmpi slt, %arg0, %arg1 : i64
    %1 = arith.select %0, %arg1, %arg0 : i64
    return %1, %arg8 : i64, none
}

handshake.func @mux(%select : i1, %a : i64, %b : i64) -> i64{
  %0 = handshake.mux %select [%a, %b] : i1, i64
  return %0 : i64
}

handshake.func @test_conditional_branch(%arg0: i1, %arg1: index, %arg2: none, ...) -> (index, index, none) {
  %0:2 = cond_br %arg0, %arg1 : index
  return %0#0, %0#1, %arg2 : index, index, none
}

handshake.func @test_constant(%arg0: none) -> (i32) {
  %1 = constant %arg0 {value = 42 : i32} : i32
  return %1: i32
}
