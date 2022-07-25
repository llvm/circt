// RUN: circt-opt %s -split-input-file -test-infer-top-level -verify-diagnostics | FileCheck %s

// CHECK: module attributes {test.top = "baz"}
module {
  hw.module @bar() -> () {}
  hw.module @foo() -> () {
    hw.instance "bar" @bar() -> ()
  }
  
  hw.module @baz() -> () {
    hw.instance "foo" @foo() -> ()
  }
}

// -----

// Test cycle through a component
// expected-error @+1 {{'builtin.module' op cannot deduce top level module - cycle detected in instance graph (bar->baz->foo->bar).}}
module {
  hw.module @bar() -> () {
    hw.instance "baz" @baz() -> ()
  }

  hw.module @foo() -> () {
    hw.instance "bar" @bar() -> ()
  }
  
  hw.module @baz() -> () {
    hw.instance "foo" @foo() -> ()
  }
}

// -----

// test multiple candidate top components
// expected-error @+1 {{'builtin.module' op multiple candidate top-level modules detected (bar, foo).}}
module {
  hw.module @bar() -> () {
    hw.instance "baz" @baz() -> ()
  }

  hw.module @foo() -> () {
    hw.instance "baz" @baz() -> ()
  }
  
  hw.module @baz() -> () {}
}
