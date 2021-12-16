// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK: firrtl.module @innerFIFO_2_ui32(
// CHECK: firrtl.module @handshake_buffer_in_ui32_out_ui32_2slots_fifo(
handshake.func @test_buffer_p2(%arg0: i32, %arg1: none, ...) -> (i32, none) {
  %0 = buffer [2] %arg0 {sequential = false} : i32
  return %0, %arg1 : i32, none
}

// -----

// CHECK: firrtl.module @innerFIFO_5_ui32(
// CHECK: firrtl.module @handshake_buffer_in_ui32_out_ui32_5slots_fifo(
handshake.func @test_buffer_unequal(%arg0: i32, %arg1: none, ...) -> (i32, none) {
  %0 = buffer [5] %arg0 {sequential = false} : i32
  return %0, %arg1 : i32, none
}

// -----

// CHECK: firrtl.module @innerFIFO_3_ctrl(
// CHECK: firrtl.module @handshake_buffer_3slots_fifo_1ins_1outs_ctrl(
handshake.func @test_fifo_nonetype(%arg0: none, %arg1: none, ...) -> (none, none) {
  %0 = buffer [3] %arg0 {sequential = false} : none
  return %0, %arg1 : none, none
}
