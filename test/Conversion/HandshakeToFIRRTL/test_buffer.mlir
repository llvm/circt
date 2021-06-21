// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_buffer_1ins_1outs_ctrl_3slots_seq(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
// CHECK:   %[[IN_VALID:.+]] = firrtl.subfield %arg0("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %[[IN_READY:.+]] = firrtl.subfield %arg0("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %[[OUT_VALID:.+]] = firrtl.subfield %arg1("valid") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %[[OUT_READY:.+]] = firrtl.subfield %arg1("ready") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>>) -> !firrtl.uint<1>
// CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>

// Stage 0 ready wire and valid register.
// CHECK:   %readyWire0 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %validReg0 = firrtl.regreset %clock, %reset, %c0_ui1 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %ctrlValidWire0 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %ctrlReadyWire0 = firrtl.wire : !firrtl.uint<1>

// pred_ready = !reg_valid || succ_ready.
// CHECK:   %[[VAL_4:.+]] = firrtl.not %validReg0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_5:.+]] = firrtl.or %[[VAL_4:.+]], %readyWire0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %[[IN_READY:.+]], %[[VAL_5:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// Drive valid register.
// CHECK:   %[[VAL_6:.+]] = firrtl.mux(%[[VAL_5:.+]], %[[IN_VALID:.+]], %validReg0) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %validReg0, %[[VAL_6:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// Stage 0 ready register.
// CHECK:   %readyReg = firrtl.regreset %clock, %reset, %c0_ui1_0

// succ_valid = readyReg ? readyReg : pred_valid
// CHECK:   %[[SUCC_VALID0:.+]] = firrtl.mux(%readyReg, %readyReg, %validReg0)
// CHECK:   firrtl.connect %ctrlValidWire0, %[[SUCC_VALID0]]

// pred_ready = !readyReg
// CHECK:   %[[NOT_READY0:.+]] = firrtl.not %readyReg
// CHECK:   firrtl.connect %readyWire0, %[[NOT_READY0]]

// Stage 1 logics.
// CHECK:   %readyWire1 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %validReg1 = firrtl.regreset %clock, %reset, %c0_ui1 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %ctrlValidWire1 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %ctrlReadyWire1 = firrtl.wire : !firrtl.uint<1>

// CHECK:   %[[VAL_7:.+]] = firrtl.not %validReg1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_8:.+]] = firrtl.or %[[VAL_7:.+]], %readyWire1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %ctrlReadyWire0, %[[VAL_8:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %[[VAL_9:.+]] = firrtl.mux(%[[VAL_8:.+]], %ctrlValidWire0, %validReg1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %validReg1, %[[VAL_9:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// Stage 1 ready register.
// CHECK:   %[[READY_REG1:.+]] = firrtl.regreset %clock, %reset, %c0_ui1_1 {name = "readyReg"}

// succ_valid = readyReg ? readyReg : pred_valid
// CHECK:   %[[SUCC_VALID1:.+]] = firrtl.mux(%[[READY_REG1]], %[[READY_REG1]], %validReg1)
// CHECK:   firrtl.connect %ctrlValidWire1, %[[SUCC_VALID1]]

// pred_ready = !readyReg
// CHECK:   %[[NOT_READY1:.+]] = firrtl.not %[[READY_REG1]]
// CHECK:   firrtl.connect %readyWire1, %[[NOT_READY1]]

// Stage 2 logics.
// CHECK:   %readyWire2 = firrtl.wire : !firrtl.uint<1>
// CHECK:   %validReg2 = firrtl.regreset %clock, %reset, %c0_ui1 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

// CHECK:   %[[VAL_10:.+]] = firrtl.not %validReg2 : (!firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   %[[VAL_11:.+]] = firrtl.or %[[VAL_10:.+]], %readyWire2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %ctrlReadyWire1, %[[VAL_11:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// CHECK:   %[[VAL_12:.+]] = firrtl.mux(%[[VAL_11:.+]], %ctrlValidWire1, %validReg2) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
// CHECK:   firrtl.connect %validReg2, %[[VAL_12:.+]] : !firrtl.uint<1>, !firrtl.uint<1>

// Connet to output ports.
// CHECK:   firrtl.connect %[[OUT_VALID:.+]], %ctrlValidWire2 : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK:   firrtl.connect %ctrlReadyWire2, %[[OUT_READY:.+]] : !firrtl.uint<1>, !firrtl.uint<1>
// CHECK: }

// CHECK-LABEL: firrtl.module @test_buffer(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>,
// CHECK-SAME:  in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_buffer(%arg0: none, %arg1: none, ...) -> (none, none) {

  // CHECK: %inst_arg0, %inst_arg1, %inst_clock, %inst_reset = firrtl.instance @handshake_buffer_1ins_1outs_ctrl_3slots_seq {name = ""} : !firrtl.flip<bundle<valid: uint<1>, ready flip: uint<1>>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, !firrtl.flip<clock>, !firrtl.flip<uint<1>>
  %0 = "handshake.buffer"(%arg0) {control = true, sequential = true, slots = 3 : i32} : (none) -> none
  handshake.return %0, %arg1 : none, none
}

// -----

// CHECK-LABEL: firrtl.module @handshake_buffer_1ins_1outs_ui64_2slots_seq(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
// CHECK:   %[[IN_DATA:.+]] = firrtl.subfield %arg0("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %[[OUT_DATA:.+]] = firrtl.subfield %arg1("data") : (!firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) -> !firrtl.uint<64>
// CHECK:   %c0_ui64 = firrtl.constant 0 : !firrtl.uint<64>

// CHECK:   %dataReg0 = firrtl.regreset %clock, %reset, %c0_ui64 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<64>) -> !firrtl.uint<64>

// CHECK:   %[[VAL_9:.+]] = firrtl.mux(%[[VAL_7:.+]], %[[IN_DATA:.+]], %dataReg0) : (!firrtl.uint<1>, !firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %dataReg0, %[[VAL_9:.+]] : !firrtl.uint<64>, !firrtl.uint<64>

// CHECK:   %ctrlDataReg = firrtl.regreset %clock, %reset, %c0_ui64_1
// CHECK:   %[[SUCC_DATA0:.+]] = firrtl.mux(%readyReg{{.*}}, %ctrlDataReg, %dataReg0)
// CHECK:   firrtl.connect %ctrlDataWire0, %[[SUCC_DATA0]]

// CHECK:   %dataReg1 = firrtl.regreset %clock, %reset, %c0_ui64 : (!firrtl.clock, !firrtl.uint<1>, !firrtl.uint<64>) -> !firrtl.uint<64>

// CHECK:   %[[CTRL_DATA_REG1:.+]] = firrtl.regreset %clock, %reset, %c0_ui64_6 {name = "ctrlDataReg"}
// CHECK:   %[[SUCC_DATA1:.+]] = firrtl.mux(%readyReg{{.*}}, %[[CTRL_DATA_REG1]], %dataReg1)
// CHECK:   firrtl.connect %ctrlDataWire1, %[[SUCC_DATA1]]

// CHECK:   %[[VAL_13:.+]] = firrtl.mux(%[[VAL_11:.+]], %dataReg1, %[[CTRL_DATA_REG1]]) : (!firrtl.uint<1>, !firrtl.uint<64>, !firrtl.uint<64>) -> !firrtl.uint<64>
// CHECK:   firrtl.connect %ctrlDataRegWire{{.*}}, %[[VAL_13:.+]] : !firrtl.uint<64>, !firrtl.uint<64>


// CHECK:   firrtl.connect %[[OUT_DATA:.+]], %ctrlDataWire1 : !firrtl.uint<64>, !firrtl.uint<64>
// CHECK: }

// CHECK-LABEL: firrtl.module @test_buffer_data(
// CHECK-SAME:  in %arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in %arg1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out %arg2: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
handshake.func @test_buffer_data(%arg0: index, %arg1: none, ...) -> (index, none) {

  // CHECK: %inst_arg0, %inst_arg1, %inst_clock, %inst_reset = firrtl.instance @handshake_buffer_1ins_1outs_ui64_2slots_seq {name = ""} : !firrtl.flip<bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>>, !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, !firrtl.flip<clock>, !firrtl.flip<uint<1>>
  // CHECK: firrtl.connect %inst_clock, %clock : !firrtl.flip<clock>, !firrtl.clock
  // CHECK: firrtl.connect %inst_reset, %reset : !firrtl.flip<uint<1>>, !firrtl.uint<1>
  %0 = "handshake.buffer"(%arg0) {control = false, sequential = true, slots = 2 : i32} : (index) -> index
  handshake.return %0, %arg1 : index, none
}
