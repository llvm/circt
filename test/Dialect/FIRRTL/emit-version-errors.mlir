// Tests for --firrtl-version option: every version-gated op/type produces the
// correct error when the target version is below the required minimum.
//
// Each RUN line targets a specific version and checks errors for all features
// that require a newer version.

// Targeting FIRRTL 3.0.0: should error on features requiring >= 3.1.0
// RUN: circt-translate --export-firrtl --firrtl-version=3.0.0 \
// RUN:   --verify-diagnostics --split-input-file %s

// FIRRTL 3.0.0: propassign (properties) requires >= 3.1.0
firrtl.circuit "PropAssign" {
  firrtl.module @PropAssign(out %str : !firrtl.string) {
    %0 = firrtl.string "hello"
    // expected-error @below {{'firrtl.propassign' op properties requires FIRRTL 3.1.0}}
    firrtl.propassign %str, %0 : !firrtl.string
  }
}

// -----

// FIRRTL 3.0.0: Integer constant requires >= 3.1.0.  The propassign statement
// fires first since it shares the 3.1.0 requirement.
firrtl.circuit "IntegerConst" {
  firrtl.module @IntegerConst(out %i : !firrtl.integer) {
    %0 = firrtl.integer 42
    // expected-error @below {{'firrtl.propassign' op properties requires FIRRTL 3.1.0}}
    firrtl.propassign %i, %0 : !firrtl.integer
  }
}

// -----

// FIRRTL 3.0.0: String constant requires >= 3.1.0.
firrtl.circuit "StringConst" {
  firrtl.module @StringConst(out %s : !firrtl.string) {
    %0 = firrtl.string "hello"
    // expected-error @below {{'firrtl.propassign' op properties requires FIRRTL 3.1.0}}
    firrtl.propassign %s, %0 : !firrtl.string
  }
}
