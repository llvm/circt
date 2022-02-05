// RUN: circt-opt %s -verify-diagnostics -split-input-file

hw.module.extern @Foo()

hw.module @Top() {
  // expected-error @+1 {{Unknown device type 'WACKY'}}
  hw.instance "foo1" @Foo() -> () {"loc:" = #msft.physloc<WACKY, 0, 0, 0> } 
}

// -----

module {
  // expected-error @+1 {{unknown attribute `foo` in dialect `msft`}}
  hw.instance "foo1" @Foo() -> () {"loc:" = #msft.foo<""> } 
}

// -----

hw.module @M() {
  // expected-error @+1 {{Cannot find module definition 'Bar'}}
  msft.instance @instance @Bar () : () -> ()
}
