// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-types)' %s -split-input-file -verify-diagnostics

firrtl.circuit "Uniquification" {
  // expected-error@below {{port names should be unique}}
  firrtl.module @Uniquification(in %a: !firrtl.bundle<b: uint<1>>, in %a_b: !firrtl.uint<1>) {
  }
}
