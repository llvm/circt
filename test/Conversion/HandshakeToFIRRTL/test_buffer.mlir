// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_buffer_1ins_1outs_ctrl_3slots_seq(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
// CHECK:   %[[IN_VALID:.+]] = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.uint<1>
// CHECK:   %[[IN_READY:.+]] = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>) -> !firrtl.flip<uint<1>>
// CHECK:   %[[OUT_VALID:.+]] = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.flip<uint<1>>
// CHECK:   %[[OUT_READY:.+]] = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %c0_ui1 = firrtl.constant(0 : ui1) : !firrtl.uint<1>

// Stage 0 ready wire and valid register.
// CHECK:   %readyWire0 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %validReg0 = firrtl.regreset %clock, %reset, %c0_ui1 {name = "validReg0"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

// pred_ready = !reg_valid || succ_ready.
// CHECK:   %[[VAL_4:.+]] = firrtl.not %validReg0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_5:.+]] = firrtl.or %[[VAL_4:.+]], %readyWire0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %[[IN_READY:.+]], %[[VAL_5:.+]] : !firrtl.flip<uint<1>>, !firrtl.uint<1>

// Drive valid register.
// CHECK:   %[[VAL_6:.+]] = firrtl.mux(%[[VAL_5:.+]], %[[IN_VALID:.+]], %validReg0) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %validReg0, %[[VAL_6:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// Stage 1 logics.
// CHECK:   %readyWire1 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %validReg1 = firrtl.regreset %clock, %reset, %c0_ui1 {name = "validReg1"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

// CHECK:   %[[VAL_7:.+]] = firrtl.not %validReg1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_8:.+]] = firrtl.or %[[VAL_7:.+]], %readyWire1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %readyWire0, %[[VAL_8:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %[[VAL_9:.+]] = firrtl.mux(%[[VAL_8:.+]], %validReg0, %validReg1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %validReg1, %[[VAL_9:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// Stage 2 logics.
// CHECK:   %readyWire2 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %validReg2 = firrtl.regreset %clock, %reset, %c0_ui1 {name = "validReg2"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

// CHECK:   %[[VAL_10:.+]] = firrtl.not %validReg2 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_11:.+]] = firrtl.or %[[VAL_10:.+]], %readyWire2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %readyWire1, %[[VAL_11:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %[[VAL_12:.+]] = firrtl.mux(%[[VAL_11:.+]], %validReg1, %validReg2) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %validReg2, %[[VAL_12:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// Connet to output ports.
// CHECK:   firrtl.connect %[[OUT_VALID:.+]], %validReg2 : !firrtl.flip<uint<1>>, !firrtl.uint<1>
// CHECK:   firrtl.connect %readyWire2, %[[OUT_READY:.+]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK: }

// CHECK-LABEL: firrtl.module @test_buffer(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
handshake.func @test_buffer(%arg0: none, %arg1: none, ...) -> (none, none) {

  // CHECK: %inst_arg0, %inst_arg1, %inst_clock, %inst_reset = firrtl.instance @handshake_buffer_1ins_1outs_ctrl_3slots_seq {name = "", portNames = ["arg0", "arg1", "clock", "reset"]} : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, !firrtl.flip<clock>, !firrtl.flip<uint<1>>
  %0 = "handshake.buffer"(%arg0) {control = true, sequential = true, slots = 3 : i32} : (none) -> none
  handshake.return %0, %arg1 : none, none
}

// -----

// CHECK-LABEL: firrtl.module @handshake_buffer_1ins_1outs_ui64_2slots_seq(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
// CHECK:   %[[IN_DATA:.+]] = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %[[OUT_DATA:.+]] = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>) -> !firrtl.flip<uint<64>>
// CHECK:   %c0_ui64 = firrtl.constant(0 : ui64) : !firrtl.uint<64>

// CHECK:   %dataReg0 = firrtl.regreset %clock, %reset, %c0_ui64 {name = "dataReg0"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<64>) -> !firrtl.uint<64>
// CHECK:   %[[VAL_9:.+]] = firrtl.mux(%[[VAL_7:.+]], %[[IN_DATA:.+]], %dataReg0) : (!firrtl.uint<1>, !firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %dataReg0, %[[VAL_9:.+]] : !firrtl.uint<64>, !firrtl.uint<64>

// CHECK:   %dataReg1 = firrtl.regreset %clock, %reset, %c0_ui64 {name = "dataReg1"} : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<64>) -> !firrtl.uint<64>
// CHECK:   %[[VAL_13:.+]] = firrtl.mux(%[[VAL_11:.+]], %dataReg0, %dataReg1) : (!firrtl.uint<1>, !firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %dataReg1, %[[VAL_13:.+]] : !firrtl.uint<64>, !firrtl.uint<64>

// CHECK:   firrtl.connect %[[OUT_DATA:.+]], %dataReg1 : !firrtl.flip<uint<64>>, !firrtl.uint<64>
// CHECK: }

// CHECK-LABEL: firrtl.module @test_buffer_data(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
handshake.func @test_buffer_data(%arg0: index, %arg1: none, ...) -> (index, none) {

  // CHECK: %inst_arg0, %inst_arg1, %inst_clock, %inst_reset = firrtl.instance @handshake_buffer_1ins_1outs_ui64_2slots_seq {name = "", portNames = ["arg0", "arg1", "clock", "reset"]} : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, !firrtl.flip<clock>, !firrtl.flip<uint<1>>
  // CHECK: firrtl.connect %inst_clock, %clock : !firrtl.flip<clock>, !firrtl.clock
  // CHECK: firrtl.connect %inst_reset, %reset : !firrtl.flip<uint<1>>, !firrtl.uint<1>  
  %0 = "handshake.buffer"(%arg0) {control = false, sequential = true, slots = 2 : i32} : (index) -> index
  handshake.return %0, %arg1 : index, none
}
