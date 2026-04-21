// Tests for --firrtl-version=4.0.0:
//   1. intmodule was removed at 4.0.0 and errors when targeting >= 4.0.0.
//   2. inline layer convention requires >= 4.1.0 and errors when targeting 4.0.0.
// RUN: circt-translate --export-firrtl --firrtl-version=4.0.0 \
// RUN:   --verify-diagnostics --split-input-file %s

// intmodule was removed at FIRRTL 4.0.0
firrtl.circuit "IntModule" {
  // expected-error @below {{'firrtl.intmodule' op intrinsic modules were removed in FIRRTL 4.0.0}}
  firrtl.intmodule @Foo(in i : !firrtl.clock) attributes {intrinsic = "foo"}
  firrtl.module @IntModule() {}
}

// -----

// inline layer convention requires >= 4.1.0
firrtl.circuit "InlineLayer" {
  // expected-error @below {{'firrtl.layer' op inline layers requires FIRRTL 4.1.0}}
  firrtl.layer @A inline {}
  firrtl.module @InlineLayer() {}
}
