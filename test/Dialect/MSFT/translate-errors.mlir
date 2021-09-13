// RUN: circt-translate %s --export-quartus-tcl -verify-diagnostics -split-input-file

hw.module.extern @Foo()

// expected-error @+1 {{Could not place 1 instances}}
hw.module @top() {
  // expected-error @+1 {{PhysLoc attribute must be inside an instance switch attribute}}
  hw.instance "foo1" @Foo() {"loc:" = #msft.physloc<DSP, 0, 0, 0> } -> ()
}

// -----

hw.module.extern @Foo()

// expected-error @+1 {{Could not place 1 instances}}
hw.module @top() {
  // expected-error @+1 {{'hw.instance' op PhysLoc attributes must have names starting with 'loc'}}
  hw.instance "foo1" @Foo() {"phys:" = #msft.switch.inst< @top[] = #msft.physloc<DSP, 0, 0, 0> > } -> ()
}
