// RUN: circt-translate %s --export-quartus-tcl -verify-diagnostics -split-input-file

rtl.module.extern @Foo()

rtl.module @top() {
  // expected-error @+1 {{Entity name cannot be empty in 'loc:<entityName>'}}
  rtl.instance "foo1" @Foo() {"loc:" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
}

// -----

rtl.module.extern @Foo()

rtl.module @top() {
  // expected-error @+1 {{Error in 'phys:' PhysLocation attribute. Expected loc:<entityName>}}
  rtl.instance "foo1" @Foo() {"phys:" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
}
