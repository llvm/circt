// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK:           firrtl.module @foo(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_4:.*]]: !firrtl.clock, in %[[VAL_5:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_6:.*]], %[[VAL_7:.*]], %[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]], %[[VAL_11:.*]] = firrtl.instance bar0  @bar(in a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in ctrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out outCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             firrtl.connect %[[VAL_6]], %[[VAL_0]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_7]], %[[VAL_1]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_10]], %[[VAL_4]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_11]], %[[VAL_5]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_2]], %[[VAL_8]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_3]], %[[VAL_9]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }

// CHECK:           firrtl.module @bar(in %[[VAL_12:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_13:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_14:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_15:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_16:.*]]: !firrtl.clock, in %[[VAL_17:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_18:.*]], %[[VAL_19:.*]], %[[VAL_20:.*]], %[[VAL_21:.*]], %[[VAL_22:.*]], %[[VAL_23:.*]] = firrtl.instance baz0  @baz(in a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in ctrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out outCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             firrtl.connect %[[VAL_18]], %[[VAL_12]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_19]], %[[VAL_13]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_22]], %[[VAL_16]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_23]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_14]], %[[VAL_20]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_15]], %[[VAL_21]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }

// CHECK:           firrtl.module @handshake_fork_in_ui32_out_ui32_ui32(in %[[VAL_24:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_25:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_26:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_27:.*]]: !firrtl.clock, in %[[VAL_28:.*]]: !firrtl.uint<1>) {

// CHECK:           firrtl.module @arith_addi_in_ui32_ui32_out_ui32(in %[[VAL_29:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_30:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_31:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {

// CHECK:           firrtl.module @baz(in %[[VAL_77:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_78:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_79:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_80:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_81:.*]]: !firrtl.clock, in %[[VAL_82:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_83:.*]], %[[VAL_84:.*]], %[[VAL_85:.*]], %[[VAL_86:.*]], %[[VAL_87:.*]] = firrtl.instance handshake_fork0  @handshake_fork_in_ui32_out_ui32_ui32(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             %[[VAL_88:.*]], %[[VAL_89:.*]], %[[VAL_90:.*]] = firrtl.instance arith_addi0  @arith_addi_in_ui32_ui32_out_ui32(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in in1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>)
// CHECK:             firrtl.connect %[[VAL_83]], %[[VAL_77]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_86]], %[[VAL_81]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_87]], %[[VAL_82]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_88]], %[[VAL_84]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_89]], %[[VAL_85]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_79]], %[[VAL_90]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_80]], %[[VAL_78]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }
// CHECK:         }

module {
  handshake.func @baz(%a: i32, %ctrl : none) -> (i32, none) {
    %0:2 = "handshake.fork"(%a) {control = false} : (i32) -> (i32, i32)
    %1 = arith.addi %0#0, %0#1 : i32
    handshake.return %1, %ctrl : i32, none
  }

  handshake.func @bar(%a: i32, %ctrl : none) -> (i32, none) {
    %c:2 = handshake.instance @baz(%a, %ctrl) : (i32, none) -> (i32, none)
    handshake.return %c#0, %c#1 : i32, none
  }

  handshake.func @foo(%a: i32, %ctrl : none) -> (i32, none) {
    %b:2 = handshake.instance @bar(%a, %ctrl) : (i32, none) -> (i32, none)
    handshake.return %b#0, %b#1: i32, none
  }
}
