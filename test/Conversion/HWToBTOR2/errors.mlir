// RUN: circt-opt %s --convert-hw-to-btor2 --verify-diagnostics --split-input-file -o /dev/null

hw.module @init_emitter(out out: !seq.immutable<i32>) {
  %init = seq.initial () {
    %c0_i32 = hw.constant 0 : i32
    seq.yield %c0_i32 : i32
  } : () -> !seq.immutable<i32>
  hw.output %init : !seq.immutable<i32>
}

hw.module @reg_with_instance_initial(in %clk: !seq.clock, in %in: i32, out out: i32) {
  // expected-error @below {{'hw.instance' op not supported in BTOR2 conversion}}
  %init = hw.instance "foo" @init_emitter () -> (out: !seq.immutable<i32>)
  // expected-error @below {{Initial value must be emitted directly by a seq.initial op}}
  %1 = seq.compreg %in, %clk initial %init : i32
  hw.output %1 : i32
}

// -----

hw.module @reg_with_argument_initial(in %clk: !seq.clock, in %in: i32, in %init: !seq.immutable<i32>, out out: i32) {
  // expected-error @below {{Initial value must be emitted directly by a seq.initial op}}
  %1 = seq.compreg %in, %clk initial %init : i32
  hw.output %1 : i32
}

// -----

hw.module @dual_clock_error(in %in : i32, in %clk : !seq.clock, in %clk1 : !seq.clock) {
  %0 = seq.compreg %in, %clk : i32
// expected-error @below {{Multi-clock designs are not currently supported.}}
  %1 = seq.compreg %in, %clk1 : i32
}

// -----

hw.module @nullary_variadic() {
  // expected-error @below {{variadic operations with no operands are not supported}}
  "comb.concat"() : () -> (i0)
}

// -----

hw.module @multi_event_always(in %clk : !seq.clock) {
  %0 = seq.from_clock %clk
  // expected-error @below {{Multiple events in sv.always are not supported.}}
  sv.always posedge %0, negedge %0 {
  }
}

// -----

hw.module @multi_clk_always(in %clk : !seq.clock) {
  %0 = seq.from_clock %clk
  // expected-error @below {{Only posedge clocking is supported in sv.always.}}
  sv.always negedge %0 {
  }
}

// -----

hw.module @i1_clk_always(in %clk : i1) {
  // expected-error @below {{This pass only currently supports sv.always ops that use a top-level seq.clock input (converted using seq.from_clock) as their clock.}}
  sv.always posedge %clk {
  }
}

// -----

hw.module @multi_clk_always(in %clk : !seq.clock, in %clk1 : !seq.clock) {
  %0 = seq.from_clock %clk
  %1 = seq.from_clock %clk1
  sv.always posedge %0 {
  }
  // expected-error @below {{Multi-clock designs are not currently supported.}}
  sv.always posedge %1 {
  }
}

// -----

hw.module @from_clock_comb(in %clk : !seq.clock) {
  // expected-error @below {{This pass only supports seq.from_clock results being used by sv.always and verif.clocked_assert operations.}}
  %0 = seq.from_clock %clk
  %1 = comb.and %0, %0 : i1
}

// -----

// expected-error @below {{Inputs converted to clocks may only have one user.}}
hw.module @to_clock_and_comb(in %a : i1) {
  %0 = seq.to_clock %a
  %1 = comb.and %a, %a : i1
}

// -----

hw.module @to_clock_reuse(in %a : i1) {
  // expected-error @below {{This pass only supports seq.to_clock results being used by seq.firreg and seq.compreg operations.}}
  %0 = seq.to_clock %a
  %1 = seq.from_clock %0
}

// -----

hw.module @to_clock_reuse(in %a : i1, in %b : i1) {
  %0 = comb.and %a, %b : i1
  // expected-error @below {{This pass only supports seq.to_clock operations that take a top-level input as their argument.}}
  %1 = seq.to_clock %0
}

// -----

hw.module @to_clock_reuse(in %a : i1, in %b : i1, in %c : i32) {
  %0 = seq.to_clock %a
  %1 = seq.to_clock %b
  %2 = seq.compreg %c, %0 : i32
  // expected-error @below {{Multi-clock designs are not currently supported.}}
  %3 = seq.compreg %c, %1 : i32
}
