// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @main(
// CHECK:  in %a: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %b: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  in %ctrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  out %arg3: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<64>>, 
// CHECK:  out %arg4: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, 
// CHECK:  in %clock: !firrtl.clock, 
// CHECK:  in %reset: !firrtl.uint<1>) {
handshake.func @main(%a: index, %b: index, %ctrl: none, ...) -> (index, none) {
  %0 = arith.addi %a, %b : index
  handshake.return %0, %ctrl : index, none
}
