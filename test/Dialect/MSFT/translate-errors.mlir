// RUN: circt-opt %s --lower-msft-to-hw=tops=top -verify-diagnostics -split-input-file

hw.module.extern @Foo()

// expected-error @+1 {{Could not place 1 instances}}
msft.module @top {} () -> () {
  // expected-error @+1 {{PhysLoc attribute must be inside an instance switch attribute}}
  msft.instance @foo1 @Foo() {"loc:" = #msft.physloc<DSP, 0, 0, 0> } : () -> ()
  msft.output
}

// -----

hw.module.extern @Foo()

// expected-error @+1 {{Could not place 1 instances}}
msft.module @top {} () -> () {
  // expected-error @+1 {{'msft.instance' op PhysLoc attributes must have names starting with 'loc'}}
  msft.instance @foo1 @Foo() {"phys:" = #msft.switch.inst< @top[] = #msft.physloc<DSP, 0, 0, 0> > } : () -> ()
  msft.output
}
