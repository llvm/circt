// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-signatures))' %s --verify-diagnostics --split-input-file

firrtl.circuit "InnerSym" {
// expected-error @below {{Port ["x"] should be subdivided, but cannot be because of symbol ["x"] on a bundle}}
  firrtl.module @InnerSym(
    in %x: !firrtl.bundle<a: uint<5>, b: uint<3>> sym [<@x,0,public>]
  ) attributes {convention = #firrtl<convention scalarized>} { }
}

// -----

firrtl.circuit "InnerSymMore" {
//expected-error @below {{Port ["x"] should be subdivided, but cannot be because of symbol ["y"] on a vector}}
  firrtl.module @InnerSymMore(
    in %x: !firrtl.vector<uint<3>, 4> sym [<@y,0, public>]
  ) attributes {convention = #firrtl<convention scalarized>}  { }
}
