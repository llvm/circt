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

// -----

rtl.module.extern @Foo()

rtl.module @top() {
  rtl.instance "foo1" @Foo() {"loc:memBank1" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
  // expected-warning @+1 {{Attribute has already been emitted: 'loc:memBank1'}}
  rtl.instance "foo2" @Foo() {"loc:memBank1" = #msft.physloc<DSP, 1, 0, 0> } : () -> ()
}

// -----

rtl.module.extern @Foo()

rtl.module @bar() {
  // expected-warning @+1 {{The placement information for this module has already been emitted. Modules are required to only be instantiated once.}}
  rtl.instance "foo1" @Foo() {"loc:memBank" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
}

rtl.module @top() {
  rtl.instance "bar1" @bar() : () -> ()
  rtl.instance "bar2" @bar() : () -> ()
}
