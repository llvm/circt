// RUN: circt-opt -allow-unregistered-dialect %s | circt-opt -allow-unregistered-dialect | FileCheck %s

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
    // CHECK: %0 = "foo"() : () -> !firrtl.fstring
    %0 = "foo"() : () -> !firrtl.fstring
  }
}
