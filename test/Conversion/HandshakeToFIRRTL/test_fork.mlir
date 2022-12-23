// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_fork_1ins_2outs_ctrl(
// CHECK-SAME:  in %[[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>
// CHECK:   %[[ARG_VALID:.+]] = firrtl.subfield %[[ARG0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[ARG_READY:.+]] = firrtl.subfield %[[ARG0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[RES0_VALID:.+]] = firrtl.subfield %[[ARG1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[RES0_READY:.+]] = firrtl.subfield %[[ARG1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[RES1_VALID:.+]] = firrtl.subfield %[[ARG2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>
// CHECK:   %[[RES1_READY:.+]] = firrtl.subfield %[[ARG2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>

// Done logic.
// CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK:   %done0 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %done1 = firrtl.wire : !firrtl.uint<1>

// CHECK:   %allDone = firrtl.wire : !firrtl.uint<1>
// CHECK:   %6 = firrtl.and %done1, %done0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %allDone, %6 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %[[ARG_READY:.+]], %allDone : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %notAllDone = firrtl.wire : !firrtl.uint<1>
// CHECK:   %7 = firrtl.not %allDone : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %notAllDone, %7 : !firrtl.uint<1>, !firrtl.uint<1>

// Result 0 logic.
// CHECK:   %emtd0 = firrtl.regreset %clock, %reset, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %8 = firrtl.and %done0, %notAllDone : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %emtd0, %8 : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %notEmtd0 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %9 = firrtl.not %emtd0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %notEmtd0, %9 : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %10 = firrtl.and %notEmtd0, %[[ARG_VALID:.+]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %[[RES0_VALID:.+]], %10 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %validReady0 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %11 = firrtl.and %[[RES0_READY:.+]], %10 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %validReady0, %11 : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %12 = firrtl.or %validReady0, %emtd0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %done0, %12 : !firrtl.uint<1>, !firrtl.uint<1>

// Result1 logic.
// CHECK:   %emtd1 = firrtl.regreset %clock, %reset, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %13 = firrtl.and %done1, %notAllDone : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %emtd1, %13 : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %notEmtd1 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %14 = firrtl.not %emtd1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %notEmtd1, %14 : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %15 = firrtl.and %notEmtd1, %[[ARG0_VALID:.+]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %[[RES1_VALID:.+]], %15 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   %validReady1 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %16 = firrtl.and %[[RES1_READY:.+]], %15 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %validReady1, %16 : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %17 = firrtl.or %validReady1, %emtd1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %done1, %17 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK: }

// CHECK: firrtl.module @test_fork(in %[[VAL_34:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_35:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_36:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_37:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_38:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_39:.*]]: !firrtl.clock, in %[[VAL_40:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_41:.*]], %[[VAL_42:.*]], %[[VAL_43:.*]], %[[VAL_44:.*]], %[[VAL_45:.*]] = firrtl.instance handshake_fork0  @handshake_fork_1ins_2outs_ctrl(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
handshake.func @test_fork(%arg0: none, %arg1: none, ...) -> (none, none, none) {
  %0:2 = fork [2] %arg0 : none
  return %0#0, %0#1, %arg1 : none, none, none
}

// -----

// CHECK-LABEL: firrtl.module @handshake_fork_in_ui64_out_ui64_ui64(
// CHECK-SAME: in %[[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
// CHECK:   %[[ARG_DATA:.+]] = firrtl.subfield %[[ARG0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[RES0_DATA:.+]] = firrtl.subfield %[[ARG1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[RES1_DATA:.+]] = firrtl.subfield %[[ARG2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>

// CHECK:   firrtl.connect %[[RES0_DATA:.+]], %[[ARG_DATA:.+]] : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK:   firrtl.connect %[[RES1_DATA:.+]], %[[ARG_DATA:.+]] : !firrtl.uint<64>, !firrtl.uint<64>

// CHECK: firrtl.module @test_fork_data(in %[[VAL_37:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_38:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_39:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_40:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_41:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_42:.*]]: !firrtl.clock, in %[[VAL_43:.*]]: !firrtl.uint<1>) {
// CHECK:             %[[VAL_44:.*]], %[[VAL_45:.*]], %[[VAL_46:.*]], %[[VAL_47:.*]], %[[VAL_48:.*]] = firrtl.instance handshake_fork0  @handshake_fork_in_ui64_out_ui64_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
handshake.func @test_fork_data(%arg0: index, %arg1: none, ...) -> (index, index, none) {
  %0:2 = fork [2] %arg0 : index
  return %0#0, %0#1, %arg1 : index, index, none
}
