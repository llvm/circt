// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include firrtl-constantifier | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Simple"
firrtl.circuit "Simple" {
  // CHECK: firrtl.module @Simple
  firrtl.module @Simple() {
    // Don't touch existing constants.
    // CHECK-NEXT: firrtl.constant 0
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

  // CHECK: firrtl.class @Properties
  firrtl.class @Properties() {
    // CHECK-NEXT: [[TMP:%.+]] = firrtl.string ""
    // CHECK-NEXT: dbg.variable "str", [[TMP]]
    %str = firrtl.string "foo"
    dbg.variable "str", %str : !firrtl.string

    // CHECK-NEXT: [[TMP:%.+]] = firrtl.integer 0
    // CHECK-NEXT: dbg.variable "int", [[TMP]]
    %int = firrtl.integer 42
    dbg.variable "int", %int : !firrtl.integer

    // CHECK-NEXT: [[TMP:%.+]] = firrtl.bool false
    // CHECK-NEXT: dbg.variable "bool", [[TMP]]
    %bool = firrtl.bool true
    dbg.variable "bool", %bool : !firrtl.bool

    // CHECK-NEXT: [[TMP:%.+]] = firrtl.double 0.0
    // CHECK-NEXT: dbg.variable "double", [[TMP]]
    %double = firrtl.double 3.141
    dbg.variable "double", %double : !firrtl.double

    // CHECK-NEXT: [[TMP:%.+]] = firrtl.list.create :
    // CHECK-NEXT: dbg.variable "list", [[TMP]]
    %list = firrtl.list.create %int, %int : !firrtl.list<integer>
    dbg.variable "list", %list : !firrtl.list<integer>

    // CHECK-NEXT: [[TMP:%.+]] = firrtl.unresolved_path ""
    // CHECK-NEXT: dbg.variable "path", [[TMP]]
    %path = firrtl.unresolved_path "foo"
    dbg.variable "path", %path : !firrtl.path
  }
}
