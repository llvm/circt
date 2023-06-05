// RUN: circt-opt --lower-firrtl-to-hw --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    %0 = firrtl.int.ltl.delay %a, 42 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @below {{operand of type '!ltl.sequence' cannot be used as an integer}}
    // expected-error @below {{couldn't handle this operation}}
    %1 = firrtl.and %0, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    %0 = firrtl.wire : !firrtl.uint<1>
    // expected-note @below {{leaking outside verification context here}}
    %1 = firrtl.and %0, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @below {{verification operation used in a non-verification context}}
    %2 = firrtl.int.ltl.delay %a, 42 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.strictconnect %0, %2 : !firrtl.uint<1>
  }
}
