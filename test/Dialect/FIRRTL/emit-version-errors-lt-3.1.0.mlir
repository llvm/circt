// Tests for --firrtl-version: ops requiring >= 3.1.0 error when targeting 3.0.0.
// RUN: circt-translate --export-firrtl --firrtl-version=3.0.0 \
// RUN:   --verify-diagnostics --split-input-file %s

// propassign (properties) requires >= 3.1.0
firrtl.circuit "PropAssign" {
  firrtl.module @PropAssign(out %str : !firrtl.string) {
    %0 = firrtl.string "hello"
    // expected-error @below {{'firrtl.propassign' op properties requires FIRRTL 3.1.0}}
    firrtl.propassign %str, %0 : !firrtl.string
  }
}
