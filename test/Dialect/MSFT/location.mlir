// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

rtl.module.extern @Foo()

rtl.module @Top() {
  // CHECK: rtl.instance "foo1" @Foo() {"loc:memBank1" = #msft.physloc<M20K, 0, 0, 0>}
  rtl.instance "foo1" @Foo() {"loc:memBank1" = #msft.physloc<M20K, 0, 0, 0> } : () -> ()
}
