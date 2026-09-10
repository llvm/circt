// RUN: circt-opt %s --msft-export-tcl=tops=top -verify-diagnostics -split-input-file

hw.module.extern @Foo()

hw.hierpath @ref [@top::@foo1]
// expected-error @+1 {{'msft.pd.physregion' op could not find physical region declaration named @region1}}
msft.pd.physregion @ref @region1

hw.module @top() {
  hw.instance "foo1" sym @foo1 @Foo() -> ()
}
