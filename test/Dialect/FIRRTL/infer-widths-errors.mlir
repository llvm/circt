// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-widths))' --verify-diagnostics --split-input-file %s

firrtl.circuit "Foo" {
  firrtl.module @Foo(in %clk: !firrtl.clock) {
    // expected-error @+1 {{'firrtl.reg' op is constrained to be wider than itself}}
    %0 = firrtl.reg %clk : !firrtl.uint
    // expected-note @+1 {{constrained width W >= W+3 here}}
    %1 = firrtl.shl %0, 3 : (!firrtl.uint) -> !firrtl.uint
    // expected-note @+1 {{constrained width W >= W+4 here}}
    %2 = firrtl.shl %1, 1 : (!firrtl.uint) -> !firrtl.uint
    // expected-note @+1 {{constrained width W >= 2W+4 here}}
    %3 = firrtl.mul %0, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    // expected-note @+1 {{constrained width W >= 2W+4 here}}
    firrtl.connect %0, %3 : !firrtl.uint, !firrtl.uint
  }
}

// -----
firrtl.circuit "Foo" {
  // expected-note @+1 {{Module `Bar` defined here:}}
  firrtl.extmodule @Bar(in in: !firrtl.uint, out out: !firrtl.uint)
  firrtl.module @Foo(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    // expected-error @+4 {{extern module `Bar` has ports of uninferred width}}
    // expected-note @+3 {{Port: "in"}}
    // expected-note @+2 {{Port: "out"}}
    // expected-note @+1 {{Only non-extern FIRRTL modules may contain unspecified widths to be inferred automatically.}}
    %inst_in, %inst_out = firrtl.instance inst @Bar(in in: !firrtl.uint, out out: !firrtl.uint)
    firrtl.connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    firrtl.connect %out, %inst_out : !firrtl.uint, !firrtl.uint
  }
}

// -----
firrtl.circuit "Issue1255" {
  // CHECK-LABEL: @Issue1255
  firrtl.module @Issue1255() {
    %tmp74 = firrtl.wire  : !firrtl.uint
    %c14972_ui = firrtl.constant 14972 : !firrtl.uint
    // expected-error @+1 {{amount must be less than or equal operand width}}
    %0 = firrtl.tail %c14972_ui, 15 : (!firrtl.uint) -> !firrtl.uint
    firrtl.connect %tmp74, %0 : !firrtl.uint, !firrtl.uint
  }
}

// -----
firrtl.circuit "Foo" {
  // expected-error @+1 {{uninferred width: port "a" is unconstrained}}
  firrtl.module @Foo (in %a: !firrtl.uint) {
  }
}

// -----
firrtl.circuit "Foo" {
  firrtl.module @Foo () {
    // expected-error @+1 {{uninferred width: wire is unconstrained}}
    %0 = firrtl.wire : !firrtl.uint
  }
}

// -----
firrtl.circuit "Foo" {
  firrtl.module @Foo () {
    // expected-error @+1 {{uninferred width: wire field "[0]" is unconstrained}}
    %0 = firrtl.wire : !firrtl.vector<uint, 16>
  }
}

// -----
firrtl.circuit "Foo" {
  firrtl.module @Foo () {
    // expected-error @+1 {{uninferred width: wire field "a" is unconstrained}}
    %0 = firrtl.wire : !firrtl.bundle<a: uint>
  }
}

// -----
firrtl.circuit "Foo" {
  firrtl.module @Foo () {
    // expected-error @+1 {{uninferred width: wire field "a.b.c" is unconstrained}}
    %0 = firrtl.wire : !firrtl.bundle<a: bundle<b: bundle<c flip: uint, d: uint<1>>>>
  }
}

// -----
// Only first error on module ports reported.
firrtl.circuit "Foo" {
  // expected-error @+2 {{uninferred width: port "a" is unconstrained}}
  // expected-error @+1 {{uninferred width: port "b" is unconstrained}}
  firrtl.module @Foo (in %a: !firrtl.uint, in %b: !firrtl.sint) {
  }
}

// -----
firrtl.circuit "Foo" {
  // expected-error @+2 {{uninferred width: port "a" is unconstrained}}
  // expected-error @+1 {{uninferred width: port "b" width cannot be determined}}
  firrtl.module @Foo(in %a: !firrtl.uint, out %b: !firrtl.uint) {
    // expected-note @+1 {{width is constrained by an uninferred width here:}}
    firrtl.connect %b, %a : !firrtl.uint, !firrtl.uint
  }
}

// -----
firrtl.circuit "Foo"  {
  firrtl.module @Foo() {
    // expected-error @+1 {{uninferred width: invalid value is unconstrained}}
    %0 = firrtl.invalidvalue : !firrtl.bundle<x: uint>
  }
}

// -----
firrtl.circuit "Foo"  {
  firrtl.module @Foo() {
    // This should complain about the wire, not the invalid value.
    %0 = firrtl.invalidvalue : !firrtl.bundle<x: uint>
    // expected-error @+1 {{uninferred width: wire "w.x" is unconstrained}}
    %w = firrtl.wire  : !firrtl.bundle<x: uint>
    firrtl.connect %w, %0 : !firrtl.bundle<x: uint>, !firrtl.bundle<x: uint>
  }
}
