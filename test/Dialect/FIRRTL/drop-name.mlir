// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl.module(firrtl-drop-names))' %s | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo() {
    // CHECK-NEXT:  %a = firrtl.wire  : !firrtl.uint<1>
    %a = firrtl.wire : !firrtl.uint<1>
  }
}
