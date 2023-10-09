// RUN: circt-opt %s -verify-diagnostics --split-input-file

hw.module @top(in %clk : !seq.clock, in %rst: i1, in %i: i32) {
  // expected-error@+2 {{expected ','}}
  // expected-error@+1 {{'seq.compreg' expected reset and resetValue operands}}
  seq.compreg %i, %clk reset %rst : i32
}

// -----

hw.module @top(in %clk : !seq.clock, in %rst: i1, in %i: i32) {
  // expected-error@+2 {{expected ','}}
  // expected-error@+1 {{'seq.compreg' expected input and clock operands}}
  seq.compreg %i : i32
}

// -----


hw.module @top(in %clk : !seq.clock, in %rst: i1, in %i: i32) {
  // expected-error@+2 {{expected SSA operand}}
  // expected-error@+1 {{'seq.compreg' expected input and clock operands}}
  seq.compreg : i32
}

// -----

hw.module @top(in %clk : !seq.clock, in %rst: i1, in %i: i32) {
  %rv = hw.constant 0 : i32
  // expected-error@+1 {{expected ':'}}
  seq.compreg %i, %clk reset %rst, %rv, %rv : i32
}

// -----
hw.module @top_ce(in %clk : !seq.clock, in %rst: i1, in %ce: i1, in %i: i32) {
  // expected-error@+2 {{expected ','}}
  // expected-error@+1 {{'seq.compreg.ce' expected reset and resetValue operands}}
  %r0 = seq.compreg.ce %i, %clk, %ce reset %rst : i32
}

// -----

hw.module @top_ce(in %clk : !seq.clock, in %rst: i1, in %ce: i1, in %i: i32) {
  // expected-error@+2 {{expected ','}}
  // expected-error@+1 {{'seq.compreg.ce' expected clock enable operand}}
  %r0 = seq.compreg.ce %i, %clk : i32
}

// -----

hw.module @top_powerOn(in %clk : !seq.clock, in %rst : i1, in %powerOn : i32, in %i : i32) {
  %rv = hw.constant 0 : i32
  // expected-error@+2 {{expected SSA operand}}
  // expected-error@+1 {{'seq.compreg' expected powerOn operand}}
  %r0 = seq.compreg %i, %clk reset %rst, %rv powerOn : i32
}
