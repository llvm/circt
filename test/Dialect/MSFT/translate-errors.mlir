// RUN: circt-translate %s --export-quartus-tcl -verify-diagnostics -split-input-file

hw.module.extern @Foo()

hw.module @top() {
  // expected-error @+1 {{Entity name cannot be empty in 'loc:<entityName>'}}
  hw.instance "foo1" @Foo() {"loc:" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
}

// -----

hw.module.extern @Foo()

hw.module @top() {
  // expected-error @+1 {{Error in 'phys:' PhysLocation attribute. Expected loc:<entityName>}}
  hw.instance "foo1" @Foo() {"phys:" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
}

// -----

hw.module.extern @Foo()

hw.module @top() {
  hw.instance "foo1" @Foo() {"loc:memBank1" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
  // expected-warning @+1 {{Attribute has already been emitted: 'loc:memBank1'}}
  hw.instance "foo2" @Foo() {"loc:memBank1" = #msft.physloc<DSP, 1, 0, 0> } : () -> ()
}

// -----

hw.module.extern @Foo()

hw.module @bar() {
  // expected-warning @+1 {{The placement information for this module has already been emitted. Modules are required to only be instantiated once.}}
  hw.instance "foo1" @Foo() {"loc:memBank" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
}

hw.module @top() {
  hw.instance "bar1" @bar() : () -> ()
  hw.instance "bar2" @bar() : () -> ()
}
