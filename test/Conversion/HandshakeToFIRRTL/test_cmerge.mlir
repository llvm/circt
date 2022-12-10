// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// Test a control merge that is control only.

// CHECK:           firrtl.module @handshake_control_merge_out_ui64_2ins_2outs_ctrl(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[CLOCK:.*]]: !firrtl.clock, in %[[RESET:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[ARG0_VALID:.+]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[ARG0_READY:.+]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[ARG1_VALID:.+]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[ARG1_READY:.+]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[ARG2_VALID:.+]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[ARG2_READY:.+]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[ARG3_VALID:.+]] = firrtl.subfield %[[VAL_3]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG3_READY:.+]] = firrtl.subfield %[[VAL_3]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG3_DATA:.+]] = firrtl.subfield %[[VAL_3]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>

// Common definitions.
// CHECK:   %[[NO_WINNER:.+]] = firrtl.constant 0 : !firrtl.uint<2>
// CHECK:   %[[FALSE_CONST:.+]] = firrtl.constant 0 : !firrtl.uint<1>

// Won register and win wire.
// CHECK:   %[[WON:won]] = firrtl.regreset %[[CLOCK]], %[[RESET]], %[[NO_WINNER]] {{.+}} !firrtl.uint<2>
// CHECK:   %win = firrtl.wire : !firrtl.uint<2>

// Fired wire, emitted registers, and done wires.
// CHECK:   %fired = firrtl.wire : !firrtl.uint<1>
// CHECK:   %[[RESULT_EMITTED:resultEmitted]] = firrtl.regreset %[[CLOCK]], %[[RESET]], %[[FALSE_CONST]] {{.+}} !firrtl.uint<1>
// CHECK:   %[[CONTROL_EMITTED:controlEmitted]] = firrtl.regreset %[[CLOCK]], %[[RESET]], %[[FALSE_CONST]] {{.+}} !firrtl.uint<1>
// CHECK:   %[[RESULT_DONE:resultDone]] = firrtl.wire : !firrtl.uint<1>
// CHECK:   %[[CONTROL_DONE:controlDone]] = firrtl.wire : !firrtl.uint<1>

// Common conditions.
// CHECK:   %[[HAS_WINNER:.+]] = firrtl.orr %win
// CHECK:   %[[HAD_WINNER:.+]] = firrtl.orr %[[WON]]

// Arbiter logic to assign win wire.
// CHECK:   %[[INDEX1:.+]] = firrtl.constant 2
// CHECK:   %[[ARB1:.+]] = firrtl.mux(%[[ARG1_VALID]], %[[INDEX1]], %[[NO_WINNER]])
// CHECK:   %[[INDEX0:.+]] = firrtl.constant 1
// CHECK:   %[[ARB0:.+]] = firrtl.mux(%[[ARG0_VALID]], %[[INDEX0]], %[[ARB1]])
// CHECK:   %[[ARB_RESULT:.+]] = firrtl.mux(%[[HAD_WINNER]], %[[WON]], %[[ARB0]])
// CHECK:   firrtl.connect %win, %[[ARB_RESULT]]

// Logic to assign result and control outputs.
// CHECK:   %[[RESULT_NOT_EMITTED:.+]] = firrtl.not %[[RESULT_EMITTED]]
// CHECK:   %[[RESULT_VALID0:.+]] = firrtl.and %[[HAS_WINNER]], %[[RESULT_NOT_EMITTED]]
// CHECK:   firrtl.connect %[[ARG2_VALID]], %[[RESULT_VALID0]]
// CHECK:   %[[CONTROL_NOT_EMITTED:.+]] = firrtl.not %[[CONTROL_EMITTED]]
// CHECK:   %[[CONTROL_VALID0:.+]] = firrtl.and %[[HAS_WINNER]], %[[CONTROL_NOT_EMITTED]]
// CHECK:   firrtl.connect %[[ARG3_VALID]], %[[CONTROL_VALID0]]

// CHECK:   %[[C0:.+]] = firrtl.constant 0
// CHECK:   %[[C1:.+]] = firrtl.constant 1
// CHECK:   %[[DEFAULT1:.+]] = firrtl.constant 0
// CHECK:   %[[BITS2:.+]] = firrtl.bits %win 1 to 1
// CHECK:   %[[CONNECT_VALID0:.+]] = firrtl.mux(%[[BITS2]], %[[C1]], %[[DEFAULT1]])
// CHECK:   %[[BITS3:.+]] = firrtl.bits %win 0 to 0
// CHECK:   %[[CONNECT_VALID1:.+]] = firrtl.mux(%[[BITS3]], %[[C0]], %[[CONNECT_VALID0]])
// CHECK:   firrtl.connect %[[ARG3_DATA]], %[[CONNECT_VALID1]]

// Logic to assign won register.
// CHECK:   %[[WON_RESULT:.+]] = firrtl.mux(%fired, %[[NO_WINNER]], %win)
// CHECK:   firrtl.connect %[[WON]], %[[WON_RESULT]]

// Logic to assign done wires.
// CHECK:   %[[RESULT_DONE0:.+]] = firrtl.and %[[RESULT_VALID0]], %[[ARG2_READY]]
// CHECK:   %[[RESULT_DONE1:.+]] = firrtl.or %[[RESULT_EMITTED]], %[[RESULT_DONE0]]
// CHECK:   firrtl.connect %[[RESULT_DONE]], %[[RESULT_DONE1]]
// CHECK:   %[[CONTROL_DONE0:.+]] = firrtl.and %[[CONTROL_VALID0]], %[[ARG3_READY]]
// CHECK:   %[[CONTROL_DONE1:.+]] = firrtl.or %[[CONTROL_EMITTED]], %[[CONTROL_DONE0]]
// CHECK:   firrtl.connect %[[CONTROL_DONE]], %[[CONTROL_DONE1]]

// Logic to assign fired wire.
// CHECK:   %[[FIRED0:.+]] = firrtl.and %[[RESULT_DONE]], %[[CONTROL_DONE]]
// CHECK:   firrtl.connect %fired, %[[FIRED0]]

// Logic to assign emitted registers.
// CHECK:   %[[RESULT_EMITTED0:.+]] = firrtl.mux(%fired, %[[FALSE_CONST]], %[[RESULT_DONE]])
// CHECK:   firrtl.connect %[[RESULT_EMITTED]], %[[RESULT_EMITTED0]]
// CHECK:   %[[CONTROL_EMITTED0:.+]] = firrtl.mux(%fired, %[[FALSE_CONST]], %[[CONTROL_DONE]])
// CHECK:   firrtl.connect %[[CONTROL_EMITTED]], %[[CONTROL_EMITTED0]]

// Logic to assign arg ready outputs.
// CHECK:   %[[WIN_OR_DEFAULT:.+]] = firrtl.mux(%fired, %win, %[[NO_WINNER]])
// CHECK:   %[[ARG0_READY0:.+]] = firrtl.eq %[[WIN_OR_DEFAULT]], %[[INDEX0]]
// CHECK:   firrtl.connect %[[ARG0_READY]], %[[ARG0_READY0]]
// CHECK:   %[[ARG1_READY0:.+]] = firrtl.eq %[[WIN_OR_DEFAULT]], %[[INDEX1]]
// CHECK:   firrtl.connect %[[ARG1_READY]], %[[ARG1_READY0]]

// CHECK:           firrtl.module @test_cmerge(in %[[VAL_53:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_54:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_55:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_56:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_57:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_58:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_59:.*]]: !firrtl.clock, in %[[VAL_60:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_61:.*]], %[[VAL_62:.*]], %[[VAL_63:.*]], %[[VAL_64:.*]], %[[VAL_65:.*]], %[[VAL_66:.*]] = firrtl.instance handshake_control_merge0  @handshake_control_merge_out_ui64_2ins_2outs_ctrl(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG3:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
handshake.func @test_cmerge(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, index, none) {
  %0:2 = control_merge %arg0, %arg1 : none
  return %0#0, %0#1, %arg2 : none, index, none
}

// -----

// Test a control merge that also outputs the selected input's data.

// CHECK:           firrtl.module @handshake_control_merge_in_ui64_ui64_out_ui64_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_3:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_4:.*]]: !firrtl.clock, in %[[VAL_5:.*]]: !firrtl.uint<1>) {
// CHECK: %[[ARG0_DATA:.+]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: %[[ARG1_DATA:.+]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK: %[[ARG2_DATA:.+]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// ...
// CHECK:   %win = firrtl.wire : !firrtl.uint<2>
// ...
// CHECK:   %[[DEFAULT0:.+]] = firrtl.constant 0
// CHECK:   %[[BITS0:.+]] = firrtl.bits %win 1 to 1
// CHECK:   %[[RESULT_DATA0:.+]] = firrtl.mux(%[[BITS0]], %[[ARG1_DATA]], %[[DEFAULT0]])
// CHECK:   %[[BITS1:.+]] = firrtl.bits %win 0 to 0
// CHECK:   %[[RESULT_DATA1:.+]] = firrtl.mux(%[[BITS1]], %[[ARG0_DATA]], %[[RESULT_DATA0]])
// CHECK:   firrtl.connect %[[ARG2_DATA]], %[[RESULT_DATA1]]
handshake.func @test_cmerge_data(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, index, none) {
  %0:2 = control_merge %arg0, %arg1 : index
  return %0#0, %0#1, %arg2 : index, index, none
}
