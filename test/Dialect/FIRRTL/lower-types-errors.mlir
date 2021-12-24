// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-lower-types)' --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %a : !firrtl.bundle<a: uint<42>>, in %b : !firrtl.bundle<a: uint<42>, b: uint<9001>>) {
    // expected-warning @+1 {{symbol @sym1 dropped after lowering}}
    %x = firrtl.node sym @sym1 %a : !firrtl.bundle<a: uint<42>>
    // expected-warning @+1 {{symbol @sym2 dropped after lowering}}
    %y = firrtl.node sym @sym2 %b : !firrtl.bundle<a: uint<42>, b: uint<9001>>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo() {
    // expected-warning @+1 {{symbol @sym1 dropped after lowering}}
    %x = firrtl.wire sym @sym1 : !firrtl.bundle<a: uint<42>>
    // expected-warning @+1 {{symbol @sym2 dropped after lowering}}
    %y = firrtl.wire sym @sym2 : !firrtl.bundle<a: uint<42>, b: uint<9001>>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %clock: !firrtl.clock) {
    // expected-warning @+1 {{symbol @sym1 dropped after lowering}}
    %x = firrtl.reg sym @sym1 %clock : !firrtl.bundle<a: uint<42>>
    // expected-warning @+1 {{symbol @sym2 dropped after lowering}}
    %y = firrtl.reg sym @sym2 %clock : !firrtl.bundle<a: uint<42>, b: uint<9001>>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(
    in %a: !firrtl.bundle<a: uint<42>>,
    in %b: !firrtl.bundle<a: uint<42>, b: uint<9001>>,
    in %clock: !firrtl.clock,
    in %reset: !firrtl.asyncreset
  ) {
    // expected-warning @+1 {{symbol @sym1 dropped after lowering}}
    %x = firrtl.regreset sym @sym1 %clock, %reset, %a : !firrtl.asyncreset, !firrtl.bundle<a: uint<42>>, !firrtl.bundle<a: uint<42>>
    // expected-warning @+1 {{symbol @sym2 dropped after lowering}}
    %y = firrtl.regreset sym @sym2 %clock, %reset, %b : !firrtl.asyncreset, !firrtl.bundle<a: uint<42>, b: uint<9001>>, !firrtl.bundle<a: uint<42>, b: uint<9001>>
  }
}

// -----

firrtl.circuit "Foo" {
  firrtl.module @Foo(
    in %a: !firrtl.bundle<a: uint<42>>,
    in %b: !firrtl.bundle<a: uint<42>, b: uint<9001>>,
    in %clock: !firrtl.clock,
    in %reset: !firrtl.asyncreset
  ) {
    // expected-warning @+1 {{symbol @sym1 dropped after lowering}}
    %x_port = firrtl.mem sym @sym1 Undefined {depth = 8 : i64, name = "x", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint<42>>>
    // expected-warning @+1 {{symbol @sym2 dropped after lowering}}
    %y_port = firrtl.mem sym @sym2 Undefined {depth = 8 : i64, name = "y", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint<42>, b: uint<9001>>>
  }
}
