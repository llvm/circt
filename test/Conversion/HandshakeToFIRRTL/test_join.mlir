// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL:   firrtl.circuit "test_join"   {
// CHECK:           firrtl.module @handshake_join_2ins_1outs_ctrl(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) {
// CHECK:             %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_5:.*]] = firrtl.subfield %[[VAL_1]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_7:.*]] = firrtl.subfield %[[VAL_2]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_8:.*]] = firrtl.subfield %[[VAL_2]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_9:.*]] = firrtl.and %[[VAL_5]], %[[VAL_3]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_7]], %[[VAL_9]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             %[[VAL_10:.*]] = firrtl.and %[[VAL_8]], %[[VAL_9]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_4]], %[[VAL_10]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_6]], %[[VAL_10]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }
// CHECK:           firrtl.module @test_join(in %[[VAL_11:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_12:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_13:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_14:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_15:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_16:.*]]: !firrtl.clock, in %[[VAL_17:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]] = firrtl.instance handshake_join  @handshake_join_2ins_1outs_ctrl(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>)
handshake.func @test_join(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, none) {
  %0 = "handshake.join"(%arg0, %arg1) {control = true}: (none, none) -> none
  handshake.return %0, %arg2 : none, none
}
