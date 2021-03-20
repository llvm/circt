// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

rtl.module.extern @Foo()

rtl.module @Top() {
  // CHECK:rtl.instance "foo1" @Foo() {loc1 = {Entity = "memBank1", Num = 0 : ui64, Type = 1 : i32, X = 0 : ui64, Y = 0 : ui64}}
  rtl.instance "foo1" @Foo() {loc1 = #msft.physloc<M20K, 0, 0, 0, "memBank1"> } : () -> ()
}
