// RUN: circt-opt %s --lower-scf-to-calyx -split-input-file -verify-diagnostics

func @main() {
  br ^bb1
^bb1:
  br ^bb2
^bb2:
  // expected-error @+1 {{CFG backedge detected. Loops must be raised to 'scf.while' or 'scf.for' operations.}}
  br ^bb1
}
