// RUN: circt-opt %s --split-input-file --firrtl-link-circuits="base-circuit=Foo" --verify-diagnostics

// expected-error @-3 {{has colliding symbol A which cannot be merged}}
firrtl.circuit "Foo" {
  firrtl.module @Foo() {}
  firrtl.layer @A bind {}
}

firrtl.circuit "Foo" {
  firrtl.extmodule @Foo()
  // expected-error @+1 {{layer convention mismatch with existing layer}}
  firrtl.layer @A inline {}
}

// -----

// expected-error @-2 {{has colliding symbol Bar which cannot be merged}}
firrtl.circuit "Foo" {
  // expected-error @+1 {{'firrtl.extmodule' op declares known layers that are not defined in the linked circuit: @A}}
  firrtl.extmodule @Bar() attributes {knownLayers = [@A]}
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
  }
  firrtl.layer @A inline {}
}

firrtl.circuit "Bar" {
  firrtl.module @Bar() {}
}

// -----

// expected-error @-2 {{has colliding symbol Bar which cannot be merged}}
firrtl.circuit "Foo" {
  // expected-error @+1 {{'firrtl.extmodule' op declares known layers that are not defined in the linked circuit: @B, @C}}
  firrtl.extmodule @Bar() attributes {knownLayers = [@A, @B, @C]}
  firrtl.module @Foo() {
    firrtl.instance bar @Bar()
  }
  firrtl.layer @A bind {}
  firrtl.layer @B bind {}
  firrtl.layer @C bind {}
}

firrtl.circuit "Bar" {
  firrtl.module @Bar() {}
  firrtl.layer @A bind {}
}
