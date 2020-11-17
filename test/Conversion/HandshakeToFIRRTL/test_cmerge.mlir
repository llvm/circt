// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_control_merge_2ins_2outs_ctrl(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %[[CLOCK:.+]]: !firrtl.clock, %[[RESET:.+]]: !firrtl.uint<1>) {
// CHECK:   %[[ARG0_VALID:.+]] = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.uint<1>
// CHECK:   %1 = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %[[ARG1_VALID:.+]] = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.uint<1>
// CHECK:   %3 = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %4 = firrtl.subfield %arg2("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.flip<uint<1>>
// CHECK:   %5 = firrtl.subfield %arg2("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %6 = firrtl.subfield %arg3("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %7 = firrtl.subfield %arg3("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.uint<1>
// CHECK:   %8 = firrtl.subfield %arg3("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>

// Won register and win wire.

// CHECK:   %[[NO_WINNER:.+]] = firrtl.constant(-1 : si2) : !firrtl.sint<2>
// CHECK:   %[[WON:won]] = firrtl.reginit %[[CLOCK]], %[[RESET]], %[[NO_WINNER]] {{.+}} -> !firrtl.sint<2>
// CHECK:   %[[WIN:win]] = firrtl.wire {{.*}} : !firrtl.sint<2>

// Fired wire, emitted registers, and done wires.

// CHECK:   %[[FIRED:fired]] = firrtl.wire {{.*}} : !firrtl.uint<1>
// CHECK:   %[[EMITTED_INIT:.+]] = firrtl.constant(0 : ui1) : !firrtl.uint<1>
// CHECK:   %[[RESULT_EMITTED:resultEmitted]] = firrtl.reginit %[[CLOCK]], %[[RESET]], %[[EMITTED_INIT]] {{.+}} -> !firrtl.uint<1>
// CHECK:   %[[CONTROL_EMITTED:controlEmitted]] = firrtl.reginit %[[CLOCK]], %[[RESET]], %[[EMITTED_INIT]] {{.+}} -> !firrtl.uint<1>
// CHECK:   %[[RESULT_DONE:resultDone]] = firrtl.wire {{.*}} : !firrtl.uint<1>
// CHECK:   %[[CONTROL_DONE:controlDone]] = firrtl.wire {{.*}} : !firrtl.uint<1>

// Arbiter logic to assign win wire.

// CHECK:   %[[INDEX1:.+]] = firrtl.constant(1 : si2)
// CHECK:   %[[ARB1:.+]] = firrtl.mux(%[[ARG1_VALID]], %[[INDEX1]], %[[NO_WINNER]])
// CHECK:   %[[INDEX0:.+]] = firrtl.constant(0 : si2)
// CHECK:   %[[ARB0:.+]] = firrtl.mux(%[[ARG0_VALID]], %[[INDEX0]], %[[ARB1]])
// CHECK:   %[[HAD_WINNER:.+]] = firrtl.neq %[[WON]], %[[NO_WINNER]]
// CHECK:   %[[ARB_RESULT:.+]] = firrtl.mux(%[[HAD_WINNER]], %[[WON]], %[[ARB0]])
// CHECK:   firrtl.connect %[[WIN]], %[[ARB_RESULT]]

// Logic to assign won register.

// CHECK:   %[[WON_RESULT:.+]] = firrtl.mux(%[[FIRED]], %[[NO_WINNER]], %[[WIN]])
// CHECK:   firrtl.connect %[[WON]], %[[WON_RESULT]]

// CHECK-LABEL: firrtl.module @test_cmerge(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg2: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %arg4: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %arg5: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
handshake.func @test_cmerge(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, index, none) {

  // CHECK: %0 = firrtl.instance @handshake_control_merge_2ins_2outs_ctrl {name = ""} : !firrtl.bundle<arg0: bundle<valid: flip<uint<1>>, ready: uint<1>>, arg1: bundle<valid: flip<uint<1>>, ready: uint<1>>, arg2: bundle<valid: uint<1>, ready: flip<uint<1>>>, arg3: bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, arg4: flip<clock>, arg5: flip<uint<1>>>
  %0:2 = "handshake.control_merge"(%arg0, %arg1) {control = true} : (none, none) -> (none, index)
  handshake.return %0#0, %0#1, %arg2 : none, index, none
}
