// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include node-symbol-remover | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.module @Foo
  // CHECK-NEXT: %oneWire = firrtl.wire :
  // CHECK-NEXT: %anotherWire = firrtl.node %oneWire
  firrtl.module @Foo() {
    %oneWire = firrtl.wire sym @sym1 : !firrtl.uint<1>
    %anotherWire = firrtl.node sym @sym2 %oneWire : !firrtl.uint<1>
  }
}
