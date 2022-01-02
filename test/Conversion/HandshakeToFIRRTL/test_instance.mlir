// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK:           firrtl.module @foo(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_4:.*]]: !firrtl.clock, in %[[VAL_5:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_6:.*]], %[[VAL_7:.*]], %[[VAL_8:.*]], %[[VAL_9:.*]], %[[VAL_10:.*]] = firrtl.instance handshake_fork0  @handshake_fork_1ins_2outs_ctrl(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             %[[VAL_11:.*]], %[[VAL_12:.*]], %[[VAL_13:.*]], %[[VAL_14:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]] = firrtl.instance bar0  @bar(in a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in ctrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out outCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             %[[VAL_17:.*]] = firrtl.instance handshake_sink0  @handshake_sink_1ins_0outs_ctrl(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>)
// CHECK:             firrtl.connect %[[VAL_6]], %[[VAL_1]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_9]], %[[VAL_4]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_10]], %[[VAL_5]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_11]], %[[VAL_0]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_12]], %[[VAL_8]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_15]], %[[VAL_4]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_16]], %[[VAL_5]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_17]], %[[VAL_14]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_2]], %[[VAL_13]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_3]], %[[VAL_7]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }

// CHECK:           firrtl.module @handshake_fork_1ins_2outs_ctrl(in %[[VAL_18:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_19:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_20:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_21:.*]]: !firrtl.clock, in %[[VAL_22:.*]]: !firrtl.uint<1>) {
// CHECK:           firrtl.module @handshake_sink_1ins_0outs_ctrl(in %[[VAL_13:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) {

// CHECK:           firrtl.module @bar(in %[[VAL_55:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_56:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_57:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_58:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_59:.*]]: !firrtl.clock, in %[[VAL_60:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_61:.*]], %[[VAL_62:.*]], %[[VAL_63:.*]], %[[VAL_64:.*]], %[[VAL_65:.*]] = firrtl.instance handshake_fork0  @handshake_fork_1ins_2outs_ctrl(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             %[[VAL_66:.*]], %[[VAL_67:.*]], %[[VAL_68:.*]], %[[VAL_69:.*]], %[[VAL_70:.*]], %[[VAL_71:.*]] = firrtl.instance baz0  @baz(in a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in ctrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out outCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             %[[VAL_72:.*]] = firrtl.instance handshake_sink0  @handshake_sink_1ins_0outs_ctrl(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>)
// CHECK:             firrtl.connect %[[VAL_61]], %[[VAL_56]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_64]], %[[VAL_59]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_65]], %[[VAL_60]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_66]], %[[VAL_55]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_67]], %[[VAL_63]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_70]], %[[VAL_59]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_71]], %[[VAL_60]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_72]], %[[VAL_69]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:             firrtl.connect %[[VAL_57]], %[[VAL_68]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_58]], %[[VAL_62]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }

// CHECK:           firrtl.module @arith_addi_in_ui32_ui32_out_ui32(in %[[VAL_110:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_111:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_112:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>) {

// CHECK:           firrtl.module @baz(in %[[VAL_126:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[VAL_127:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_128:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[VAL_129:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_130:.*]]: !firrtl.clock, in %[[VAL_131:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_132:.*]], %[[VAL_133:.*]], %[[VAL_134:.*]], %[[VAL_135:.*]], %[[VAL_136:.*]] = firrtl.instance handshake_fork0  @handshake_fork_in_ui32_out_ui32_ui32(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             %[[VAL_137:.*]], %[[VAL_138:.*]], %[[VAL_139:.*]] = firrtl.instance arith_addi0  @arith_addi_in_ui32_ui32_out_ui32(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in in1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>)
// CHECK:             firrtl.connect %[[VAL_132]], %[[VAL_126]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_135]], %[[VAL_130]] : !firrtl.clock, !firrtl.clock
// CHECK:             firrtl.connect %[[VAL_136]], %[[VAL_131]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:             firrtl.connect %[[VAL_137]], %[[VAL_133]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_138]], %[[VAL_134]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_128]], %[[VAL_139]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:             firrtl.connect %[[VAL_129]], %[[VAL_127]] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:           }

module {
  handshake.func @baz(%a: i32, %ctrl : none) -> (i32, none) {
    %0:2 = fork [2] %a : i32
    %1 = arith.addi %0#0, %0#1 : i32
    return %1, %ctrl : i32, none
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
