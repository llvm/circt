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
