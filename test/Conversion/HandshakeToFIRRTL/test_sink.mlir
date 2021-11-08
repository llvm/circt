// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s


// CHECK-LABEL:   firrtl.circuit "test_sink"   {
// CHECK:           firrtl.module @handshake_sink_in_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:             %[[VAL_1:.*]] = firrtl.subfield %[[VAL_0]](1) : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:             %[[VAL_2:.*]] = firrtl.constant 1 : !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_1]], %[[VAL_2]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:           }
// CHECK:           firrtl.module @test_sink(in %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_5:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_6:.*]]: !firrtl.clock, in %[[VAL_7:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_8:.*]] = firrtl.instance handshake_sink  @handshake_sink_in_ui64(in arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
// CHECK:             firrtl.connect %[[VAL_8]], %[[VAL_3]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:             firrtl.connect %[[VAL_5]], %[[VAL_4]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }
// CHECK:         }
handshake.func @test_sink(%arg0: index, %arg1: none, ...) -> (none) {
  "handshake.sink"(%arg0) : (index) -> ()
  handshake.return %arg1 : none
}
