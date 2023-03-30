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

handshake.func @taskPipelining(%arg0: i64, %arg1: none, ...) -> (i64, none) attributes {argNames = ["in0", "in1"], resNames = ["out0", "out1"]} {
  %0 = constant %arg1 {value = 0 : i64} : i64
  %1 = arith.cmpi eq, %0, %arg0 : i64
  %2 = buffer [2] fifo %1 : i1
  %trueResult, %falseResult = cond_br %1, %arg0 : i64
  %trueResult_0, %falseResult_1 = cond_br %1, %arg1 : none
  %trueResult_2, %falseResult_3 = cond_br %1, %0 : i64
  %3 = buffer [1] seq %9 {initValues = [0]} : i1
  %4 = mux %3 [%trueResult_0, %falseResult_7] : i1, none
  %5 = mux %3 [%trueResult_2, %11] : i1, i64
  %6 = constant %4 {value = 100 : i64} : i64
  %7 = arith.cmpi eq, %6, %5 : i64
  %trueResult_4, %falseResult_5 = cond_br %7, %5 : i64
  %8 = constant %4 {value = true} : i1
  %9 = arith.xori %7, %8 : i1
  %trueResult_6, %falseResult_7 = cond_br %7, %4 : none
  %10 = constant %falseResult_7 {value = 1 : i64} : i64
  %11 = arith.addi %falseResult_5, %10 : i64
  %12 = mux %16 [%trueResult_4, %falseResult] : index, i64
  %13 = mux %2 [%falseResult_1, %trueResult_6] : i1, none
  %14 = constant %13 {value = true} : i1
  %15 = arith.xori %2, %14 : i1
  %16 = arith.index_cast %15 : i1 to index
  return %12, %13 : i64, none
}
