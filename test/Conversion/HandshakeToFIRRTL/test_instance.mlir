// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK:           firrtl.module @foo(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_4:.*]]: !firrtl.clock, in %[[VAL_5:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_6:.*]], %[[VAL_7:.*]], %[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]] = firrtl.instance bar0  @bar(in a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in ctrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out outCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             %[[VAL_12:.*]] = firrtl.instance handshake_sink0  @handshake_sink_1ins_0outs_ctrl(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>)
// CHECK:             firrtl.connect %[[VAL_6]], %[[VAL_0]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_7]], %[[VAL_1]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_10]], %[[VAL_4]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_11]], %[[VAL_5]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_12]], %[[VAL_9]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_2]], %[[VAL_8]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_3]], %[[VAL_1]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }

// CHECK:           firrtl.module @handshake_sink_1ins_0outs_ctrl(in %[[VAL_13:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) {

// CHECK:           firrtl.module @bar(in %[[VAL_16:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_17:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_18:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_19:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_20:.*]]: !firrtl.clock, in %[[VAL_21:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_22:.*]], %[[VAL_23:.*]], %[[VAL_24:.*]], %[[VAL_25:.*]], %[[VAL_26:.*]], %[[VAL_27:.*]] = firrtl.instance baz0  @baz(in a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in ctrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out outCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             %[[VAL_28:.*]] = firrtl.instance handshake_sink0  @handshake_sink_1ins_0outs_ctrl(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>)
// CHECK:             firrtl.connect %[[VAL_22]], %[[VAL_16]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_23]], %[[VAL_17]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_26]], %[[VAL_20]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_27]], %[[VAL_21]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_28]], %[[VAL_25]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_18]], %[[VAL_24]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_19]], %[[VAL_17]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }

// CHECK:           firrtl.module @arith_addi_in_ui32_ui32_out_ui32(in %[[VAL_29:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_30:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_31:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {

// CHECK:           firrtl.module @baz(in %[[VAL_45:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_46:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_47:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_48:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_49:.*]]: !firrtl.clock, in %[[VAL_50:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_51:.*]], %[[VAL_52:.*]], %[[VAL_53:.*]] = firrtl.instance arith_addi0  @arith_addi_in_ui32_ui32_out_ui32(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>)
// CHECK:             firrtl.connect %[[VAL_51]], %[[VAL_45]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_52]], %[[VAL_45]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_47]], %[[VAL_53]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_48]], %[[VAL_46]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }

module {
  handshake.func @baz(%a: i32, %ctrl : none) -> (i32, none) {
    %0 = arith.addi %a, %a : i32
    return %0, %ctrl : i32, none
  }

  handshake.func @bar(%a: i32, %ctrl : none) -> (i32, none) {
    %c, %ctrlOut = handshake.instance @baz(%a, %ctrl) : (i32, none) -> (i32, none)
    sink %ctrlOut : none
    return %c, %ctrl : i32, none
  }

  handshake.func @foo(%a: i32, %ctrl : none) -> (i32, none) {
    %b:2 = handshake.instance @bar(%a, %ctrl) : (i32, none) -> (i32, none)
    sink %b#1 : none
    return %b#0, %ctrl : i32, none
  }
}
