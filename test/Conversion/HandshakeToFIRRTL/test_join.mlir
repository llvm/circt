// RUN: circt-opt -split-input-file -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL:   firrtl.circuit "test_join"   {
// CHECK:           firrtl.module @handshake_join_2ins_1outs_ctrl(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) {
// CHECK:             %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_9:.*]] = firrtl.and %[[VAL_5]], %[[VAL_3]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_7]], %[[VAL_9]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_10:.*]] = firrtl.and %[[VAL_8]], %[[VAL_9]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_4]], %[[VAL_10]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_6]], %[[VAL_10]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }
// CHECK:           firrtl.module @test_join(in %[[VAL_11:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_12:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_13:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_14:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_15:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_16:.*]]: !firrtl.clock, in %[[VAL_17:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]] = firrtl.instance handshake_join0  @handshake_join_2ins_1outs_ctrl(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>)
handshake.func @test_join(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, none) {
  %0 = join %arg0, %arg1 : none, none
  return %0, %arg2 : none, none
}

// -----

// CHECK-LABEL:   firrtl.module @handshake_join_in_ui32_ui1_3ins_1outs_ctrl(
// CHECK-SAME:      in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>,
// CHECK-SAME:      in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>,
// CHECK-SAME:      in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:      out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) {
// CHECK:             %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_6:.*]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>
// CHECK:             %[[VAL_9:.*]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<1>>
// CHECK:             %[[VAL_10:.*]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_11:.*]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_12:.*]] = firrtl.subfield %[[VAL_3]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_13:.*]] = firrtl.subfield %[[VAL_3]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             %[[VAL_14:.*]] = firrtl.and %[[VAL_7]], %[[VAL_4]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_15:.*]] = firrtl.and %[[VAL_10]], %[[VAL_14]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_12]], %[[VAL_15]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_16:.*]] = firrtl.and %[[VAL_13]], %[[VAL_15]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_5]], %[[VAL_16]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_8]], %[[VAL_16]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_11]], %[[VAL_16]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }

handshake.func @test_join_multi_types(%arg0: i32, %arg1: i1, %arg2: none, ...) -> (none) {
  %0 = join %arg0, %arg1, %arg2 : i32, i1, none
  return %0: none
}
