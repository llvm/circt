// RUN: circt-opt --handshake-split-merges %s | FileCheck %s

// CHECK-LABEL:   handshake.func @cm4(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, ...) -> (i32, index)
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_5:.*]], %[[VAL_6:.*]] = control_merge %[[VAL_0]], %[[VAL_1]] : i32, index
// CHECK:           %[[VAL_7:.*]] = pack %[[VAL_5]], %[[VAL_6]] : tuple<i32, index>
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = control_merge %[[VAL_2]], %[[VAL_3]] : i32, index
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_4]] : index
// CHECK:           %[[VAL_11:.*]] = pack %[[VAL_8]], %[[VAL_10]] : tuple<i32, index>
// CHECK:           %[[VAL_12:.*]] = merge %[[VAL_7]], %[[VAL_11]] : tuple<i32, index>
// CHECK:           %[[VAL_13:.*]]:2 = unpack %[[VAL_12]] : tuple<i32, index>
// CHECK:           return %[[VAL_13]]#0, %[[VAL_13]]#1 : i32, index
// CHECK:         }
handshake.func @cm4(%in0 : i32, %in1 : i32, %in2 : i32, %in3 : i32) -> (i32, index) {
    %d0, %idx0 = handshake.control_merge %in0, %in1, %in2, %in3 : i32, index
    return %d0, %idx0 : i32, index
}

// CHECK-LABEL:   handshake.func @cm5(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, ...) -> (i32, index)
// CHECK:           %[[VAL_5:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]], %[[VAL_8:.*]] = control_merge %[[VAL_0]], %[[VAL_1]] : i32, index
// CHECK:           %[[VAL_9:.*]] = pack %[[VAL_7]], %[[VAL_8]] : tuple<i32, index>
// CHECK:           %[[VAL_10:.*]], %[[VAL_11:.*]] = control_merge %[[VAL_2]], %[[VAL_3]] : i32, index
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_6]] : index
// CHECK:           %[[VAL_13:.*]] = pack %[[VAL_10]], %[[VAL_12]] : tuple<i32, index>
// CHECK:           %[[VAL_14:.*]] = pack %[[VAL_4]], %[[VAL_5]] : tuple<i32, index>
// CHECK:           %[[VAL_15:.*]] = merge %[[VAL_9]], %[[VAL_13]] : tuple<i32, index>
// CHECK:           %[[VAL_16:.*]] = merge %[[VAL_15]], %[[VAL_14]] : tuple<i32, index>
// CHECK:           %[[VAL_17:.*]]:2 = unpack %[[VAL_16]] : tuple<i32, index>
// CHECK:           return %[[VAL_17]]#0, %[[VAL_17]]#1 : i32, index
// CHECK:         }
handshake.func @cm5(%in0 : i32, %in1 : i32, %in2 : i32, %in3 : i32, %in4 : i32) -> (i32, index) {
    %d0, %idx0 = handshake.control_merge %in0, %in1, %in2, %in3, %in4 : i32, index
    return %d0, %idx0 : i32, index
}

// CHECK-LABEL:   handshake.func @m3(
// CHECK-SAME:           %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, ...) -> i32
// CHECK:           %[[VAL_3:.*]] = merge %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_4:.*]] = merge %[[VAL_3]], %[[VAL_2]] : i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }
handshake.func @m3(%in0 : i32, %in1 : i32, %in2 : i32) -> (i32) {
    %out = handshake.merge %in0, %in1, %in2 : i32
    return %out : i32
}

// CHECK-LABEL:   handshake.func @m5(
// CHECK-SAME:             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, ...) -> i32
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_6:.*]] = merge %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = merge %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]] = merge %[[VAL_7]], %[[VAL_4]] : i32
// CHECK:           return %[[VAL_8]] : i32
// CHECK:         }
handshake.func @m5(%in0 : i32, %in1 : i32, %in2 : i32, %in3 : i32, %in4 : i32) -> (i32) {
    %out = handshake.merge %in0, %in1, %in2, %in3, %in4 : i32
    return %out : i32
}
