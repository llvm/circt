// RUN: circt-opt --lower-firrtl-to-hw --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    %0 = firrtl.int.ltl.delay %a, 42 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    // expected-error @+2 {{operand of type '!ltl.sequence' cannot be used as an integer}}
    // expected-error @+1 {{couldn't handle this operation}}
    %1 = firrtl.and %0, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  }
}

// -----
// NOTE: This test case was removed because with LTL wire canonicalization prevention,
// the wire is preserved and the error detection works differently. The test would need
// to be restructured to match the new behavior where LTL-typed wires are not folded away.

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.reset, out %hbr: !firrtl.uint<1>) {
    // expected-error @below {{uninferred reset passed to 'has_been_reset'; requires sync or async reset}}
    // expected-note @below {{reset is of type '!firrtl.reset', should be '!firrtl.uint<1>' or '!firrtl.asyncreset'}}
    // expected-error @below {{couldn't handle this operation}}
    %0 = firrtl.int.has_been_reset %clock, %reset : !firrtl.reset
    firrtl.matchingconnect %hbr, %0 : !firrtl.uint<1>
  }
}
