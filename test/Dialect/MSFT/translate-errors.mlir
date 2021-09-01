// RUN: circt-translate %s --export-quartus-tcl -verify-diagnostics -split-input-file

hw.module.extern @Foo()

hw.module @top() {
  // expected-error @+1 {{PhysLoc attribute must be inside an instance switch attribute}}
  hw.instance "foo1" @Foo() {"loc:" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
}

// -----

hw.module.extern @Foo()

hw.module @top() {
  // expected-error @+1 {{Error in 'phys:' PhysLocation attribute. Expected loc:<entityName>}}
  hw.instance "foo1" @Foo() {"phys:" = #msft.switch.inst< @top[] = #msft.physloc<DSP, 0, 0, 0> > } : () -> ()
}
