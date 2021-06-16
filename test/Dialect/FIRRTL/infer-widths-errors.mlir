// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-infer-widths)' --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %clk: !firrtl.clock) {
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

// -----
firrtl.circuit "Foo" {
  // expected-note @+1 {{Module `Bar` defined here:}}
  firrtl.extmodule @Bar(in %in: !firrtl.uint, out %out: !firrtl.uint)
  firrtl.module @Foo(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    // expected-error @+2 {{extern module `Bar` has ports of uninferred width}}
    // expected-note @+1 {{Only non-extern FIRRTL modules may contain unspecified widths to be inferred automatically.}}
    %inst_in, %inst_out = firrtl.instance @Bar {name = "inst"} : !firrtl.flip<uint>, !firrtl.uint
    firrtl.connect %inst_in, %in : !firrtl.flip<uint>, !firrtl.uint<42>
    firrtl.connect %out, %inst_out : !firrtl.uint, !firrtl.uint
  }
}

