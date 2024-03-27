// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-types))' %s --verify-diagnostics --split-input-file

// Check diagnostic when attempting to lower something with symbols on it.
firrtl.circuit "InnerSym" {
  firrtl.module @InnerSym(
  // expected-error @below {{unable to lower due to symbol "x" with target not preserved by lowering}}
    in %x: !firrtl.bundle<a: uint<5>, b: uint<3>>
      sym @x
    ) { }
}

// -----

// Check diagnostic when attempting to lower something with internalPath.
firrtl.circuit "InternalPath" {
  firrtl.extmodule @InternalPath(
  // expected-error @below {{cannot lower port with internal path}}
      out x: !firrtl.probe<bundle<a: uint<5>, b: uint<3>>>
    ) attributes { internalPaths = [#firrtl.internalpath<"a.b.c">] }
}
