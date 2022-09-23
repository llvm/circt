// RUN: circt-opt -lower-handshake-to-firrtl --split-input-file %s | FileCheck %s


// CHECK-LABEL: firrtl.circuit "no_args"  {
// CHECK:         firrtl.module @no_args(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
// CHECK:         }
// CHECK:       }

handshake.func @no_args() {
  return
}

// -----

// CHECK-LABEL:  firrtl.circuit "external"  {
// CHECK:          firrtl.extmodule @external(in arg0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>, data: uint<32>>, in ctrl: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, out out0: !firrtl.bundle<valid: uint<1>, ready flip: uint<1>>, in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
// CHECK:         }
handshake.func @external(%arg0: i32, %ctrl: none, ...) -> none
