// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include firrtl-constantifier | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Simple"
firrtl.circuit "Simple" {
  // CHECK: firrtl.module @Simple
  firrtl.module @Simple() {
    // Don't touch existing constants.
    // CHECK-NEXT: firrtl.constant 1337
    %c1337_ui42 = firrtl.constant 1337 : !firrtl.uint<42>

    // Turn basic operations into constants.
    // CHECK-NEXT: [[TMP:%.+]] = firrtl.constant 0 : !firrtl.sint<43>
    // CHECK-NEXT: dbg.variable "neg", [[TMP]]
    %neg = firrtl.neg %c1337_ui42 : (!firrtl.uint<42>) -> !firrtl.sint<43>
    dbg.variable "neg", %neg : !firrtl.sint<43>

    // Don't touch operations with inner symbols.
    // CHECK-NEXT: [[TMP:%.+]] = firrtl.not
    // CHECK-NEXT: dbg.variable "not", [[TMP]]
    %not = firrtl.not %c1337_ui42 {inner_sym = @foo} : (!firrtl.uint<42>) -> !firrtl.uint<42>
    dbg.variable "not", %not : !firrtl.uint<42>
  }
}
