// RUN: circt-opt -lower-handshake-to-firrtl --split-input-file %s | FileCheck %s

// CHECK-LABEL:  firrtl.module @handshake_pack_in_ui64_ui32_out_tuple_ui64_ui32(
// CHECK-SAME:    in %[[IN0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:    in %[[IN1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>,
// CHECK-SAME:    out %[[OUT:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>) {
// CHECK:    %[[IN0_VALID:.*]] = firrtl.subfield %[[IN0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:    %[[IN0_READY:.*]] = firrtl.subfield %[[IN0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:    %[[IN0_DATA:.*]] = firrtl.subfield %[[IN0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:    %[[IN1_VALID:.*]] = firrtl.subfield %[[IN1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:    %[[IN1_READY:.*]] = firrtl.subfield %[[IN1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:    %[[IN1_DATA:.*]] = firrtl.subfield %[[IN1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:    %[[OUT_VALID:.*]] = firrtl.subfield %[[OUT]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:    %[[OUT_READY:.*]] = firrtl.subfield %[[OUT]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:    %[[OUT_DATA:.*]] = firrtl.subfield %[[OUT]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:    %[[VAL_9:.*]] = firrtl.subfield %[[OUT_DATA]][field0] : !firrtl.bundle<field0: uint<64>, field1: uint<32>>
// CHECK:    %[[VAL_10:.*]] = firrtl.subfield %[[OUT_DATA]][field1] : !firrtl.bundle<field0: uint<64>, field1: uint<32>>
// CHECK:    firrtl.connect %[[VAL_9]], %[[IN0_DATA]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:    firrtl.connect %[[VAL_10]], %[[IN1_DATA]] : !firrtl.uint<32>, !firrtl.uint<32>
// CHECK:    %[[VAL_11:.*]] = firrtl.and %[[IN1_VALID]], %[[IN0_VALID]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:    firrtl.connect %[[OUT_VALID]], %[[VAL_11]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:    %[[VAL_12:.*]] = firrtl.and %[[OUT_READY]], %[[VAL_11]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:    firrtl.connect %[[IN0_READY]], %[[VAL_12]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:    firrtl.connect %[[IN1_READY]], %[[VAL_12]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  }
// CHECK:  firrtl.module @test_pack(in %[[ARG0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[ARG1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in %[[CTRL:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[OUT:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>, out %[[OUT_CTRL:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[CLOCK:.*]]: !firrtl.clock, in %[[RESET:.*]]: !firrtl.uint<1>) {
// CHECK:    %handshake_pack0_in0, %handshake_pack0_in1, %handshake_pack0_out0 = firrtl.instance handshake_pack0  @handshake_pack_in_ui64_ui32_out_tuple_ui64_ui32(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in in1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>)

handshake.func @test_pack(%arg0: i64, %arg1: i32, %ctrl: none, ...) -> (tuple<i64, i32>, none) {
  %0 = pack %arg0, %arg1 : tuple<i64, i32>
  return %0, %ctrl : tuple<i64, i32>, none
}

// -----

// CHECK-LABEL: firrtl.module @handshake_unpack_in_tuple_ui64_ui32_out_ui64_ui32(
// CHECK-SAME:    in %[[IN:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>,
// CHECK-SAME:    out %[[OUT0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:    out %[[OUT1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>,
// CHECK-SAME:    in %[[CLOCK:.*]]: !firrtl.clock,
// CHECK-SAME:    in %[[RESET:.*]]: !firrtl.uint<1>) {
// CHECK:  %[[IN_VALID:.*]] = firrtl.subfield %[[IN]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:  %[[IN_READY:.*]] = firrtl.subfield %[[IN]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:  %[[IN_DATA:.*]] = firrtl.subfield %[[IN]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>
// CHECK:  %[[OUT0_VALID:.*]] = firrtl.subfield %[[OUT0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:  %[[OUT0_READY:.*]] = firrtl.subfield %[[OUT0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:  %[[OUT0_DATA:.*]] = firrtl.subfield %[[OUT0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:  %[[OUT1_VALID:.*]] = firrtl.subfield %[[OUT1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:  %[[OUT1_READY:.*]] = firrtl.subfield %[[OUT1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:  %[[OUT1_DATA:.*]] = firrtl.subfield %[[OUT1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>
// CHECK:  %[[VAL_9:.*]] = firrtl.subfield %[[IN_DATA]][field0] : !firrtl.bundle<field0: uint<64>, field1: uint<32>>
// CHECK:  %[[VAL_10:.*]] = firrtl.subfield %[[IN_DATA]][field1] : !firrtl.bundle<field0: uint<64>, field1: uint<32>>
// CHECK:  firrtl.connect %[[OUT0_DATA]], %[[VAL_9]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:  firrtl.connect %[[OUT1_DATA]], %[[VAL_10]] : !firrtl.uint<32>, !firrtl.uint<32>


// CHECK:  %[[VAL_11:.*]] = firrtl.constant 0 : !firrtl.uint<1>
// CHECK:  %[[VAL_12:.*]] = firrtl.wire  : !firrtl.uint<1>
// CHECK:  %[[VAL_13:.*]] = firrtl.wire  : !firrtl.uint<1>
// CHECK:  %[[VAL_14:.*]] = firrtl.wire  : !firrtl.uint<1>
// CHECK:  %[[VAL_15:.*]] = firrtl.and %[[VAL_13]], %[[VAL_12]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_14]], %[[VAL_15]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  firrtl.connect %[[IN_READY]], %[[VAL_14]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_16:.*]] = firrtl.wire  : !firrtl.uint<1>
// CHECK:  %[[VAL_17:.*]] = firrtl.not %[[VAL_14]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_16]], %[[VAL_17]] : !firrtl.uint<1>, !firrtl.uint<1>

// Result 0 logic.
// CHECK:  %[[VAL_18:.*]] = firrtl.regreset %[[CLOCK]], %[[RESET]], %[[VAL_11]]  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_19:.*]] = firrtl.and %[[VAL_12]], %[[VAL_16]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_18]], %[[VAL_19]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_20:.*]] = firrtl.wire  : !firrtl.uint<1>
// CHECK:  %[[VAL_21:.*]] = firrtl.not %emtd0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_20]], %[[VAL_21]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_22:.*]] = firrtl.and %[[VAL_20]], %[[IN_VALID]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[OUT0_VALID]], %[[VAL_22]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_23:.*]] = firrtl.wire  : !firrtl.uint<1>
// CHECK:  %[[VAL_24:.*]] = firrtl.and %[[OUT0_READY]], %[[VAL_22]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_23]], %[[VAL_24]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_25:.*]] = firrtl.or %[[VAL_23]], %[[VAL_18]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_12]], %[[VAL_25]] : !firrtl.uint<1>, !firrtl.uint<1>

// Result 1 logic.
// CHECK:  %[[VAL_26:.*]] = firrtl.regreset %[[CLOCK]], %[[RESET]], %[[VAL_11]]  : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_27:.*]] = firrtl.and %[[VAL_13]], %[[VAL_16]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_26]], %[[VAL_27]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_28:.*]] = firrtl.wire  : !firrtl.uint<1>
// CHECK:  %[[VAL_29:.*]] = firrtl.not %[[VAL_26]] : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_28]], %[[VAL_29]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_30:.*]] = firrtl.and %[[VAL_28]], %[[IN_VALID]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[OUT1_VALID]], %[[VAL_30]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_31:.*]] = firrtl.wire  : !firrtl.uint<1>
// CHECK:  %[[VAL_32:.*]] = firrtl.and %[[OUT1_READY]], %[[VAL_30]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_31]], %[[VAL_32]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:  %[[VAL_33:.*]] = firrtl.or %[[VAL_31]], %[[VAL_26]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:  firrtl.connect %[[VAL_13]], %[[VAL_33]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:}

// CHECK:firrtl.module @test_unpack(in %[[ARG0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>, in %[[CTRL:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[OUT0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[OUT1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out %[[OUT_CTRL:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[CLOCK:.*]]: !firrtl.clock, in %[[RESET:.*]]: !firrtl.uint<1>) {
// CHECK:  %[[IN:.*]], %[[OUT0:.*]], %[[OUT1:.*]], %[[CLOCK:.*]], %[[RESET:.*]] = firrtl.instance handshake_unpack0  @handshake_unpack_in_tuple_ui64_ui32_out_ui64_ui32(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: bundle<field0: uint<64>, field1: uint<32>>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out out1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)

handshake.func @test_unpack(%arg0: tuple<i64, i32>, %ctrl: none, ...) -> (i64, i32, none) {
  %0, %1 = unpack %arg0 : tuple<i64, i32>
  return %0, %1, %ctrl : i64, i32, none
}
