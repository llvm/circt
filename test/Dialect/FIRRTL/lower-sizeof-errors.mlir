// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-sizeof)), lower-firrtl-to-hw)' --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  firrtl.module @Foo(out %size: !firrtl.uint<32>) {
    // expected-note @below {{operand of type '!firrtl.uint' defined here has unknown width}}
    %unknown = firrtl.wire : !firrtl.uint
    // expected-error @below {{failed to elaborate sizeof intrinsic: unable to determine operand width}}
    %sizeof = firrtl.int.sizeof %unknown :
        (!firrtl.uint) -> !firrtl.uint<32>
    firrtl.matchingconnect %size, %sizeof : !firrtl.uint<32>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(out %size: !firrtl.uint<32>) {
    // expected-note @below {{operand of type '!firrtl.bundle<a flip: uint<1>>' defined here has unknown width}}
    %unknown = firrtl.wire : !firrtl.bundle<a flip: uint<1>>
    // expected-error @below {{failed to elaborate sizeof intrinsic: unable to determine operand width}}
    %sizeof = firrtl.int.sizeof %unknown :
        (!firrtl.bundle<a flip: uint<1>>) -> !firrtl.uint<32>
    firrtl.matchingconnect %size, %sizeof : !firrtl.uint<32>
  }
}
