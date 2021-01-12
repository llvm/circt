// RUN: circt-opt -lower-firrtl-to-rtl-module %s -verify-diagnostics  | FileCheck %s

// The firrtl.circuit should be removed, the main module name moved to an
// attribute on the module.
// CHECK-NOT: firrtl.circuit

// We should get a large header boilerplate.
// CHECK:   sv.ifdef "RANDOMIZE_GARBAGE_ASSIGN"  {
// CHECK-NEXT:   sv.verbatim "`define RANDOMIZE"
// CHECK-NEXT:  }
firrtl.circuit "CantLowerArgument" {

  // expected-error @+1 {{cannot lower this port type to RTL}}
  firrtl.module @CantLowerArgument(%arg: !firrtl.bundle<int_1: flip<uint<1>>, int_out: uint<2>>) {
  }  

  // expected-error @+1 {{unexpected operation 'func' in a firrtl.circuit}}
  func private @UnknownFunction() {
    return
  }
}
