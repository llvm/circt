// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-infer-widths)' --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %0: !firrtl.uint<4>) {
    %1 = firrtl.wire : !firrtl.uint
    // expected-error @+1 {{'firrtl.partialconnect' op not supported in width inference}}
    firrtl.partialconnect %1, %0 : !firrtl.uint, !firrtl.uint<4>
  }
}

// -----
firrtl.circuit "Foo" {
  firrtl.module @Foo(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // expected-error @+1 {{'firrtl.reg' op is constrained to be wider than itself}}
    %0 = firrtl.reg %clk : (!firrtl.clock) -> !firrtl.uint
    // expected-note @+1 {{constrained width W >= W+3 here}}
    %1 = firrtl.shl %0, 3 : (!firrtl.uint) -> !firrtl.uint
    // expected-note @+1 {{constrained width W >= W+4 here}}
    %2 = firrtl.shl %1, 1 : (!firrtl.uint) -> !firrtl.uint
    // expected-note @+1 {{constrained width W >= 2W+4 here}}
    %3 = firrtl.mul %0, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    firrtl.connect %0, %3 : !firrtl.uint, !firrtl.uint
  }
}
