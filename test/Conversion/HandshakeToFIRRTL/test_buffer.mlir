// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_buffer_1ins_1outs_ui64_2slots_seq(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {

// CHECK-LABEL: firrtl.module @test_buffer(
// CHECK-SAME:  %arg0: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, %arg1: !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>>, %arg2: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, %arg3: !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>>, %clock: !firrtl.clock, %reset: !firrtl.uint<1>) {
handshake.func @test_buffer(%arg0: index, %arg1: none, ...) -> (index, none) {

  // CHECK: %inst_arg0, %inst_arg1, %inst_clock, %inst_reset = firrtl.instance @handshake_buffer_1ins_1outs_ui64_2slots_seq {name = "", portNames = ["arg0", "arg1", "clock", "reset"]} : !firrtl.bundle<valid: flip<uint<1>>, ready: uint<1>, data: flip<uint<64>>>, !firrtl.bundle<valid: uint<1>, ready: flip<uint<1>>, data: uint<64>>, !firrtl.flip<clock>, !firrtl.flip<uint<1>>
  // CHECK: firrtl.connect %inst_clock, %clock : !firrtl.flip<clock>, !firrtl.clock
  // CHECK: firrtl.connect %inst_reset, %reset : !firrtl.flip<uint<1>>, !firrtl.uint<1>  
  %0 = "handshake.buffer"(%arg0) {control = false, sequential = true, slots = 2 : i32} : (index) -> index
  handshake.return %0, %arg1 : index, none
}
