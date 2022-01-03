// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// This is a nonsensical circuit, but it infers both sink and fork ops.

// CHECK:             %[[VAL_70:.*]], %[[VAL_71:.*]], %[[VAL_72:.*]], %[[VAL_73:.*]], %[[VAL_74:.*]] = firrtl.instance handshake_fork0  @handshake_fork_in_ui32_out_ui32_ui32(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, out out1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:             %[[VAL_79:.*]] = firrtl.instance handshake_sink0  @handshake_sink_in_ui32(in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>)
handshake.func @test_materialize(%arg0: i32, %arg1 : i1, %arg2: none) -> (i32, i32, none) {
  %true, %false = cond_br %arg1, %arg0 : i32
  return %true, %arg0, %arg2 : i32, i32, none
}