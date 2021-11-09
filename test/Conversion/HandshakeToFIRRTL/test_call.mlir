// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK: firrtl.module @main(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_4:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_5:.*]]: !firrtl.clock, in %[[VAL_6:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_7:.*]], %[[VAL_8:.*]], %[[VAL_9:.*]] = firrtl.instance handshake_call  @ext(in a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in b: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>)
// CHECK:   firrtl.connect %[[VAL_7]], %[[VAL_0]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:   firrtl.connect %[[VAL_8]], %[[VAL_1]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:   firrtl.connect %[[VAL_3]], %[[VAL_9]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:   firrtl.connect %[[VAL_4]], %[[VAL_2]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK: }
// CHECK: firrtl.extmodule @ext(in a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in b: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>)

module {
  hw.module.extern @ext(%a: !esi.channel<i32>, %b: !esi.channel<i32>) -> (out: !esi.channel<i32>)
  handshake.func @main(%a: i32, %b: i32, %ctrl : none) -> (i32, none) {
    %c = handshake.call @ext(%a, %b) : (i32, i32) -> (i32)
    handshake.return %c, %ctrl : i32, none
  }
}
