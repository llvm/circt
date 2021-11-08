// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK: firrtl.module @handshake_branch_in_ui64_out_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %[[VAL_2:.*]] = firrtl.subfield %[[VAL_0]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_3:.*]] = firrtl.subfield %[[VAL_0]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_4:.*]] = firrtl.subfield %[[VAL_0]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %[[VAL_5:.*]] = firrtl.subfield %[[VAL_1]](0) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_6:.*]] = firrtl.subfield %[[VAL_1]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_7:.*]] = firrtl.subfield %[[VAL_1]](2) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %[[VAL_5]], %[[VAL_2]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %[[VAL_3]], %[[VAL_6]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %[[VAL_7]], %[[VAL_4]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK: }
// CHECK: firrtl.module @test_branch(in %[[VAL_8:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_9:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_10:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_11:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_12:.*]]: !firrtl.clock, in %[[VAL_13:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_14:.*]], %[[VAL_15:.*]] = firrtl.instance handshake_branch  @handshake_branch_in_ui64_out_ui64(in arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
handshake.func @test_branch(%arg0: index, %arg1: none, ...) -> (index, none) {
  %0 = "handshake.branch"(%arg0) {control = false}: (index) -> index
  handshake.return %0, %arg1 : index, none
}
