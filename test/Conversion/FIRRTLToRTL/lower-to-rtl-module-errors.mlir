// RUN: circt-opt -lower-firrtl-to-rtl-module %s -verify-diagnostics  | FileCheck %s

// The firrtl.circuit should be removed, the main module name moved to an
// attribute on the module.
// CHECK-NOT: firrtl.circuit

// We should get a large header boilerplate.
// CHECK:   sv.ifdef.procedural "RANDOMIZE_GARBAGE_ASSIGN"  {
// CHECK-NEXT:   sv.verbatim "`define RANDOMIZE"
// CHECK-NEXT:  }
firrtl.circuit "UnknownFunction" {

  // expected-error @+1 {{unexpected operation 'func' in a firrtl.circuit}}
  func private @UnknownFunction() {
    return
  }

}
