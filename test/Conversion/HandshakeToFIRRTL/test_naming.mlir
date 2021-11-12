// RUN: circt-opt -split-input-file -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @main(
// CHECK:  in %a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %b: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %inCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  out %out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  out %outCtrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  in %clock: !firrtl.clock, 
// CHECK:  in %reset: !firrtl.uint<1>) {
handshake.func @main(%a: index, %b: index, %inCtrl: none, ...) -> (index, none) {
  %0 = arith.addi %a, %b : index
  handshake.return %0, %inCtrl : index, none
}

// -----

// CHECK-LABEL: firrtl.module @main(
// CHECK:  in %aTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %bTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %cTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  out %outTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  out %coutTest: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  in %clock: !firrtl.clock, 
// CHECK:  in %reset: !firrtl.uint<1>) {
handshake.func @main(%a: index, %b: index, %inCtrl: none, ...) -> (index, none) attributes {argNames = ["aTest", "bTest", "cTest"], outNames = ["outTest", "coutTest"]} {
  %0 = arith.addi %a, %b : index
  handshake.return %0, %inCtrl : index, none
}


// -----

// CHECK-LABEL: firrtl.module @handshake_mux_in_ui64_ui64_ui64_out_ui64(
// CHECK:         in %select: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:         in %in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:         in %in1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:         out %out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>) {

// CHECK-LABEL: firrtl.module @test_mux(
// CHECK: %handshake_mux_select, %handshake_mux_in0, %handshake_mux_in1, %handshake_mux_out0 = firrtl.instance handshake_mux  @handshake_mux_in_ui64_ui64_ui64_out_ui64(in select: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in in0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, in in1: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>)
handshake.func @test_mux(%arg0: index, %arg1: index, %arg2: index, %arg3: none, ...) -> (index, none) {
  %0 = "handshake.mux"(%arg0, %arg1, %arg2): (index, index, index) -> index
  handshake.return %0, %arg3 : index, none
}
