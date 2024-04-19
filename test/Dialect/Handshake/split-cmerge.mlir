// RUN: circt-opt --handshake-split-cmerge %s | FileCheck %s

// CHECK-LABEL:   handshake.func @cm4(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, ...) -> (i32, index)
// CHECK:           %[[VAL_4:.*]], %[[VAL_5:.*]] = control_merge %[[VAL_0]], %[[VAL_1]] : i32, index
// CHECK:           %[[VAL_6:.*]] = pack %[[VAL_4]], %[[VAL_5]] : tuple<i32, index>
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = control_merge %[[VAL_2]], %[[VAL_3]] : i32, index
// CHECK:           %[[VAL_9:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_8]], %[[VAL_9]] : index
// CHECK:           %[[VAL_11:.*]] = pack %[[VAL_7]], %[[VAL_10]] : tuple<i32, index>
// CHECK:           %[[VAL_12:.*]] = merge %[[VAL_6]], %[[VAL_11]] : tuple<i32, index>
// CHECK:           %[[VAL_13:.*]]:2 = unpack %[[VAL_12]] : tuple<i32, index>
// CHECK:           return %[[VAL_13]]#0, %[[VAL_13]]#1 : i32, index
// CHECK:         }
handshake.func @cm4(%in0 : i32, %in1 : i32, %in2 : i32, %in3 : i32) -> (i32, index) {
    %d0, %idx0 = handshake.control_merge %in0, %in1, %in2, %in3 : i32, index
    return %d0, %idx0 : i32, index
}

// CHECK-LABEL:   handshake.func @cm5(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, ...) -> (i32, index)
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = control_merge %[[VAL_0]], %[[VAL_1]] : i32, index
// CHECK:           %[[VAL_7:.*]] = pack %[[VAL_5]], %[[VAL_6]] : tuple<i32, index>
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = control_merge %[[VAL_2]], %[[VAL_3]] : i32, index
// CHECK:           %[[VAL_10:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : index
// CHECK:           %[[VAL_12:.*]] = pack %[[VAL_8]], %[[VAL_11]] : tuple<i32, index>
// CHECK:           %[[VAL_13:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_14:.*]] = pack %[[VAL_4]], %[[VAL_13]] : tuple<i32, index>
// CHECK:           %[[VAL_15:.*]] = merge %[[VAL_7]], %[[VAL_12]], %[[VAL_14]] : tuple<i32, index>
// CHECK:           %[[VAL_16:.*]]:2 = unpack %[[VAL_15]] : tuple<i32, index>
// CHECK:           return %[[VAL_16]]#0, %[[VAL_16]]#1 : i32, index
// CHECK:         }
handshake.func @cm5(%in0 : i32, %in1 : i32, %in2 : i32, %in3 : i32, %in4 : i32) -> (i32, index) {
    %d0, %idx0 = handshake.control_merge %in0, %in1, %in2, %in3, %in4 : i32, index
    return %d0, %idx0 : i32, index
}
