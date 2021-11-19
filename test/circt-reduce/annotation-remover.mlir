// RUN: circt-reduce %s --test %S/test.sh --test-arg cat --test-arg "%anotherWire = firrtl.wire" --keep-best=0 --include annotation-remover | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK: firrtl.module @Foo
  // CHECK: %anotherWire = firrtl.wire
  // CHECK-NOT: annotations
  firrtl.module @Foo() {
    %oneWire = firrtl.wire : !firrtl.uint<1>
    %anotherWire = firrtl.wire {annotations = [{a}]} : !firrtl.uint<1>
  }
}
