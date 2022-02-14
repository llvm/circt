// RUN: circt-opt %s --lower-msft-to-hw --msft-export-tcl=tops=top -verify-diagnostics -split-input-file

hw.module.extern @Foo()

// expected-error @+1 {{'hw.globalRef' op referenced non-existant PhysicalRegion named region1}}
hw.globalRef @ref [#hw.innerNameRef<@top::@foo1>] {
  "loc:" = #msft.physical_region_ref<@region1>
}

// expected-error @+1 {{Could not place 1 instances}}
msft.module @top {} () -> () {
  msft.instance @foo1 @Foo() {circt.globalRef = [#hw.globalNameRef<@ref>], inner_sym = "foo1"} : () -> ()
  msft.output
}
