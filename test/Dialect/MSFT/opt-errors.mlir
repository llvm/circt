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

// -----

msft.instance.dynamic [#hw.innerNameRef<@reg::@reg>] {
  // expected-error @+1 {{'msft.pd.location' op cannot both have a global ref symbol and be a child of a dynamic instance op}}
  msft.pd.location @ref FF x: 0 y: 0 n: 0
}

// -----

// expected-error @+1 {{'msft.pd.location' op must have either a global ref symbol of belong to a dynamic instance op}}
msft.pd.location FF x: 0 y: 0 n: 0
