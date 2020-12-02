// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_merge_2ins_1outs(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) {
// CHECK:   %[[ARG0_VALID:.+]] = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG0_READY:.+]] = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %[[ARG0_DATA:.+]] = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %[[ARG1_VALID:.+]] = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG1_READY:.+]] = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.flip<uint<1>>
// CHECK:   %[[ARG1_DATA:.+]] = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %[[ARG2_VALID:.+]] = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %[[ARG2_READY:.+]] = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
// CHECK:   %[[ARG2_DATA:.+]] = firrtl.subfield %arg2("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>

// Common definitions.
// CHECK:   %[[NO_WINNER:.+]] = firrtl.constant(-1 : si2) : !firrtl.sint<2>

// Win wire.
// CHECK:   %[[WIN:win]] = firrtl.wire {{.*}} : !firrtl.sint<2>

// Result done wire.
// CHECK:   %[[RESULT_DONE:resultDone]] = firrtl.wire {{.*}} : !firrtl.uint<1>

// Common conditions.
// CHECK:   %[[HAS_WINNER:.+]] = firrtl.neq %[[WIN]], %[[NO_WINNER]]

// Arbiter logic to assign win wire.
// CHECK:   %[[INDEX1:.+]] = firrtl.constant(1 : si2)
// CHECK:   %[[ARB1:.+]] = firrtl.mux(%[[ARG1_VALID]], %[[INDEX1]], %[[NO_WINNER]])
// CHECK:   %[[INDEX0:.+]] = firrtl.constant(0 : si2)
// CHECK:   %[[ARB0:.+]] = firrtl.mux(%[[ARG0_VALID]], %[[INDEX0]], %[[ARB1]])
// CHECK:   firrtl.connect %[[WIN]], %[[ARB0]]

// Logic to assign result and control outputs.
// CHECK:   %[[WIN_BITS:.+]] = firrtl.bits %[[WIN]] {{.+}} to 0 {{.+}} -> !firrtl.uint
// CHECK:   %[[WIN_UNSIGNED:.+]] = firrtl.asUInt %[[WIN_BITS]]
// CHECK:   %[[BITS0:.+]] = firrtl.bits %[[WIN_UNSIGNED]] 0 to 0
// CHECK:   %[[RESULT_VALID0:.+]] = firrtl.mux(%[[BITS0]], %[[ARG1_VALID]], %[[ARG0_VALID]])
// CHECK:   %[[RESULT_VALID1:.+]] = firrtl.and %[[RESULT_VALID0]], %[[HAS_WINNER]]
// CHECK:   firrtl.connect %[[ARG2_VALID]], %[[RESULT_VALID1]]
// CHECK:   %[[BITS1:.+]] = firrtl.bits %[[WIN_UNSIGNED]] 0 to 0
// CHECK:   %[[RESULT_DATA:.+]] = firrtl.mux(%[[BITS1]], %[[ARG1_DATA]], %[[ARG0_DATA]])
// CHECK:   firrtl.connect %[[ARG2_DATA]], %[[RESULT_DATA]]

// Logic to assign result done wire.
// CHECK:   %[[RESULT_DONE0:.+]] = firrtl.and %[[RESULT_VALID1]], %[[ARG2_READY]]
// CHECK:   firrtl.connect %[[RESULT_DONE]], %[[RESULT_DONE0]]

// Logic to assign arg ready outputs.
// CHECK:   %[[ARG0_READY0:.+]] = firrtl.eq %[[WIN]], %[[INDEX0]]
// CHECK:   %[[ARG0_READY1:.+]] = firrtl.and %[[ARG0_READY0]], %[[RESULT_DONE]]
// CHECK:   firrtl.connect %[[ARG0_READY]], %[[ARG0_READY1]]
// CHECK:   %[[ARG1_READY0:.+]] = firrtl.eq %[[WIN]], %[[INDEX1]]
// CHECK:   %[[ARG1_READY1:.+]] = firrtl.and %[[ARG1_READY0]], %[[RESULT_DONE]]
// CHECK:   firrtl.connect %[[ARG1_READY]], %[[ARG1_READY1]]

// CHECK-LABEL: firrtl.module @test_merge(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %arg4: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
handshake.func @test_merge(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {

  // CHECK: %0 = firrtl.instance @handshake_merge_2ins_1outs {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>>
  %0 = "handshake.merge"(%arg0, %arg1) : (index, index) -> index
  handshake.return %0, %arg2 : index, none
}
