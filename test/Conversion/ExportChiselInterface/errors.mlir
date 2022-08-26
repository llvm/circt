// RUN: circt-opt %s --export-chisel-interface --split-input-file --verify-diagnostics

firrtl.circuit "Foo" {
  // expected-error @+1 {{Expected reset type to be inferred for exported port}}
  firrtl.module @Foo(in %reset: !firrtl.reset) {}
}

// -----

firrtl.circuit "Foo" {
  // expected-error @+1 {{Expected width to be inferred for exported port}}
  firrtl.module @Foo(in %in: !firrtl.uint) {}
}
