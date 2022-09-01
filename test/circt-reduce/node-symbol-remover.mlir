// RUN: circt-reduce %s --test %S/test.sh --test-arg cat --test-arg "%anotherWire = firrtl.node" --keep-best=0 --include node-symbol-remover --test-must-fail | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK: firrtl.module @Foo
  // CHECK: %oneWire = firrtl.wire
  // CHECK-NEXT: %anotherWire = firrtl.node %oneWire
  firrtl.module @Foo() {
    %oneWire = firrtl.wire : !firrtl.uint<1>
    %anotherWire = firrtl.node sym @SYM %oneWire : !firrtl.uint<1>
  }
}
