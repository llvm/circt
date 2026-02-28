// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types))' %s --verify-diagnostics --split-input-file

// Check diagnostic when attempting to lower something with symbols on it.
firrtl.circuit "InnerSym" {
  firrtl.module @InnerSym(
  // expected-error @below {{unable to lower due to symbol "x" with target not preserved by lowering}}
    in %x: !firrtl.bundle<a: uint<5>, b: uint<3>>
      sym @x
    ) { }
}
