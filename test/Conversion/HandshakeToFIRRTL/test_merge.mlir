// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK: firrtl.module @handshake_merge_in_ui64_ui64_out_ui64(in %[[VAL_0:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_1:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_2:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %[[ARG0_VALID:.+]] = firrtl.subfield %[[VAL_0]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG0_READY:.+]] = firrtl.subfield %[[VAL_0]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG0_DATA:.+]] = firrtl.subfield %[[VAL_0]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG1_VALID:.+]] = firrtl.subfield %[[VAL_1]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG1_READY:.+]] = firrtl.subfield %[[VAL_1]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG1_DATA:.+]] = firrtl.subfield %[[VAL_1]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG2_VALID:.+]] = firrtl.subfield %[[VAL_2]][valid] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG2_READY:.+]] = firrtl.subfield %[[VAL_2]][ready] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
// CHECK:   %[[ARG2_DATA:.+]] = firrtl.subfield %[[VAL_2]][data] : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>

// Common definitions.
// CHECK:   %[[NO_WINNER:.+]] = firrtl.constant 0 : !firrtl.uint<2>

// Win wire.
// CHECK:   %win = firrtl.wire : !firrtl.uint<2>

// Result done wire.
// CHECK:   %resultDone = firrtl.wire : !firrtl.uint<1>

// Common conditions.
// CHECK:   %[[HAS_WINNER:.+]] = firrtl.orr %win

// Arbiter logic to assign win wire.
// CHECK:   %[[INDEX1:.+]] = firrtl.constant 2
// CHECK:   %[[ARB1:.+]] = firrtl.mux(%[[ARG1_VALID]], %[[INDEX1]], %[[NO_WINNER]])
// CHECK:   %[[INDEX0:.+]] = firrtl.constant 1
// CHECK:   %[[ARB0:.+]] = firrtl.mux(%[[ARG0_VALID]], %[[INDEX0]], %[[ARB1]])
// CHECK:   firrtl.connect %win, %[[ARB0]]

// Logic to assign result outputs.
// CHECK:   firrtl.connect %[[ARG2_VALID]], %[[HAS_WINNER]]
// CHECK:   %[[DEFAULT1:.+]] = firrtl.constant 0
// CHECK:   %[[BITS2:.+]] = firrtl.bits %win 1 to 1
// CHECK:   %[[RESULT_DATA0:.+]] = firrtl.mux(%[[BITS2]], %[[ARG1_DATA]], %[[DEFAULT1]])
// CHECK:   %[[BITS3:.+]] = firrtl.bits %win 0 to 0
// CHECK:   %[[RESULT_DATA:.+]] = firrtl.mux(%[[BITS3]], %[[ARG0_DATA]], %[[RESULT_DATA0]])
// CHECK:   firrtl.connect %[[ARG2_DATA]], %[[RESULT_DATA]]

// Logic to assign result done wire.
// CHECK:   %[[RESULT_DONE0:.+]] = firrtl.and %[[HAS_WINNER]], %[[ARG2_READY]]
// CHECK:   firrtl.connect %resultDone, %[[RESULT_DONE0]]

// Logic to assign arg ready outputs.
// CHECK:   %[[WIN_OR_DEFAULT:.+]] = firrtl.mux(%resultDone, %win, %[[NO_WINNER]])
// CHECK:   %[[ARG0_READY0:.+]] = firrtl.eq %[[WIN_OR_DEFAULT]], %[[INDEX0]]
// CHECK:   firrtl.connect %[[ARG0_READY]], %[[ARG0_READY0]]
// CHECK:   %[[ARG1_READY0:.+]] = firrtl.eq %[[WIN_OR_DEFAULT]], %[[INDEX1]]
// CHECK:   firrtl.connect %[[ARG1_READY]], %[[ARG1_READY0]]

// CHECK: firrtl.module @test_merge(in %[[VAL_29:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_30:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %[[VAL_31:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %[[VAL_32:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %[[VAL_33:.*]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %[[VAL_34:.*]]: !firrtl.clock, in %[[VAL_35:.*]]: !firrtl.uint<1>) {
// CHECK:   %[[VAL_36:.*]], %[[VAL_37:.*]], %[[VAL_38:.*]] = firrtl.instance handshake_merge0  @handshake_merge_in_ui64_ui64_out_ui64(in [[ARG0:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in [[ARG1:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out [[ARG2:.+]]: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
handshake.func @test_merge(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
  %0 = merge %arg0, %arg1 : index
  return %0, %arg2 : index, none
}
