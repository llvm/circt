// RUN: circt-opt %s --split-input-file --firrtl-link-circuits="base-circuit=Foo" --verify-diagnostics

// expected-error @-3 {{has colliding symbol A which cannot be merged}}
firrtl.circuit "Foo" {
  firrtl.module @Foo() {}
  firrtl.layer @A bind {
  }
}

firrtl.circuit "Foo" {
  firrtl.extmodule @Foo()
  // expected-error @+1 {{layer convention mismatch with existing layer}}
  firrtl.layer @A inline {
  }
}
