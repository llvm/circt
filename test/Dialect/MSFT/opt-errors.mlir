// RUN: circt-opt %s -verify-diagnostics -split-input-file

rtl.module.extern @Foo()

rtl.module @Top() {
  // expected-error @+1 {{Unknown device type 'WACKY'}}
  rtl.instance "foo1" @Foo() {"loc:" = #msft.physloc<WACKY, 0, 0, 0> } : () -> ()
}

// -----

module {
  // expected-error @+1 {{Unexpected msft attribute 'foo'}}
  rtl.instance "foo1" @Foo() {"loc:" = #msft.foo<""> } : () -> ()
}
