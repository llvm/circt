// RUN: circt-opt %s -lower-std-to-handshake -split-input-file -verify-diagnostics

// expected-error @+1 {{'handshake.func' op failed to rewrite Affine loops}}
func @doubly_nested()->() {
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 10 {}
  }
  return
}
