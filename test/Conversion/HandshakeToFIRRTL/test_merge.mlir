// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_merge_2ins_1outs_ui64(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK-SAME:  in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {
// CHECK:   %[[ARG0_VALID:.+]] = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG0_READY:.+]] = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG0_DATA:.+]] = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %[[ARG1_VALID:.+]] = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG1_READY:.+]] = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG1_DATA:.+]] = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %[[ARG2_VALID:.+]] = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG2_READY:.+]] = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG2_DATA:.+]] = firrtl.subfield %arg2("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>

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

// CHECK-LABEL: firrtl.module @test_merge(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  in %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK-SAME:  out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>,
// CHECK-SAME:  out %arg4: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_merge(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {

  // CHECK: %inst_arg0, %inst_arg1, %inst_arg2 = firrtl.instance @handshake_merge_2ins_1outs_ui64 {name = ""} : !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>
  %0 = "handshake.merge"(%arg0, %arg1) : (index, index) -> index
  handshake.return %0, %arg2 : index, none
}
