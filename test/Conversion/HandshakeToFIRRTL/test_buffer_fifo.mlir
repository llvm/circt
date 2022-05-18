// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK: firrtl.module @innerFIFO_2_ui32(
// CHECK: firrtl.module @handshake_buffer_in_ui32_out_ui32_2slots_fifo(
handshake.func @test_buffer_p2(%arg0: i32, %arg1: none, ...) -> (i32, none) {
  %0 = buffer [2] fifo %arg0 : i32
  return %0, %arg1 : i32, none
}

// -----

// CHECK: firrtl.module @innerFIFO_5_ui32(
// CHECK: firrtl.module @handshake_buffer_in_ui32_out_ui32_5slots_fifo(
handshake.func @test_buffer_unequal(%arg0: i32, %arg1: none, ...) -> (i32, none) {
  %0 = buffer [5] fifo %arg0 : i32
  return %0, %arg1 : i32, none
}

// -----

// CHECK: firrtl.module @innerFIFO_3_ctrl(
// CHECK: firrtl.module @handshake_buffer_3slots_fifo_1ins_1outs_ctrl(
handshake.func @test_fifo_nonetype(%arg0: none, %arg1: none, ...) -> (none, none) {
  %0 = buffer [3] fifo %arg0 : none
  return %0, %arg1 : none, none
}

// -----
// CHECK: firrtl.module @innerFIFO_1_tuple_ui32_ui32
// CHECK: firrtl.module @handshake_buffer_in_tuple_ui32_ui32_out_tuple_ui32_ui32_1slots_fifo
handshake.func @test_buffer_tuple_fifo(%t: tuple<i32, i32>, %arg0: none, ...) -> (tuple<i32, i32>, none) attributes {argNames = ["t", "inCtrl"], resNames = ["out0", "outCtrl"]} {
  %0 = buffer [1] fifo %t : tuple<i32, i32>
  return %0, %arg0 : tuple<i32, i32>, none
}
