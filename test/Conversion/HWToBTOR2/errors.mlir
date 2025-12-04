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

hw.module @dual_clock_error(in %clk : !seq.clock, in %clk1 : !seq.clock) {
  %clk0 = seq.from_clock %clk
// expected-error @below {{Mutli-clock designs are not supported!}}
  %clk_1 = seq.from_clock %clk1 
}

// -----

hw.module @nullary_variadic() {
  // expected-error @below {{variadic operations with no operands are not supported}}
  "comb.concat"() : () -> (i0)
}

