// RUN: circt-opt --lower-scf-to-calyx %s -split-input-file -verify-diagnostics

module {
  func @f(%arg0 : f32, %arg1 : f32) -> f32 {
    // expected-error @+1 {{failed to legalize operation 'std.addf' that was explicitly marked illegal}}
    %2 = addf %arg0, %arg1 : f32
    return %2 : f32
  }
}

// -----

// expected-error @+1 {{Module contains multiple functions, but no top level function was set. Please see --top-level-function}}
module {
  func @f1() {
    return
  }
  func @f2() {
    return
  }
}

// -----

func @main() {
  br ^bb1
^bb1:
  br ^bb2
^bb2:
  // expected-error @+1 {{CFG backedge detected. Loops must be raised to 'scf.while' or 'scf.for' operations.}}
  br ^bb1
}
