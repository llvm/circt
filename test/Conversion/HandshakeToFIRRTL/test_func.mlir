// RUN: circt-opt -lower-handshake-to-firrtl %s | FileCheck %s


// CHECK-LABEL: firrtl.circuit "no_args"  {
// CHECK:         firrtl.module @no_args(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
// CHECK:         }
// CHECK:       }

handshake.func @no_args() {
  return
}

