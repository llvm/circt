// RUN: circt-opt --lower-scf-to-calyx %s -split-input-file -verify-diagnostics

module {
  func @main(%arg0 : f32, %arg1 : f32) -> f32 {
    // expected-error @+1 {{failed to legalize operation 'std.addf' that was explicitly marked illegal}}
    %2 = addf %arg0, %arg1 : f32
    return %2 : f32
  }
}