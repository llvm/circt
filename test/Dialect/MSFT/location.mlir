// RUN: circt-opt %s

rtl.module.extern @Foo()

rtl.module @Top() {
  rtl.instance "foo1" @Foo() {"location" = #msft.pd.loc(M20k, 5, 3, 0, "memBank1") } : () -> ()
}
