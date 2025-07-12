// RUN: circt-opt --externalize-registers --split-input-file --verify-diagnostics %s

// expected-error @below {{modules with multiple clocks not yet supported}}
hw.module @two_clks(in %clk0: !seq.clock, in %clk1: !seq.clock, in %in: i32, out out: i32) {
  %1 = seq.compreg %in, %clk0 : i32
  %2 = seq.compreg %1, %clk1 : i32
  hw.output %2 : i32
}

// -----

hw.module @two_clks(in %clk_i1: i1, in %in: i32, out out: i32) {
  %clk = seq.to_clock %clk_i1
  // expected-error @below {{only clocks directly given as block arguments are supported}}
  %1 = seq.compreg %in, %clk : i32
  hw.output %1 : i32
}

// -----

hw.module @reg_with_indirect_initial(in %clk: !seq.clock, in %in: i32, out out: i32) {
  %init = seq.initial () {
    %c0_i32 = hw.constant 0 : i32
    %sum = comb.add %c0_i32, %c0_i32 : i32
    seq.yield %sum : i32
  } : () -> !seq.immutable<i32>

  // expected-error @below {{registers with initial values not directly defined by a hw.constant op in a seq.initial op not yet supported}}
  %1 = seq.compreg %in, %clk initial %init : i32
  hw.output %1 : i32
}

// -----

hw.module @reg_with_argument_initial(in %clk: !seq.clock, in %in: i32, in %init: !seq.immutable<i32>, out out: i32) {
  // expected-error @below {{registers with initial values not directly defined by a seq.initial op not yet supported}}
  %1 = seq.compreg %in, %clk initial %init : i32
  hw.output %1 : i32
}

// -----

hw.module @init_emitter(out out: !seq.immutable<i32>) {
  %init = seq.initial () {
    %c0_i32 = hw.constant 0 : i32
    seq.yield %c0_i32 : i32
  } : () -> !seq.immutable<i32>
  hw.output %init : !seq.immutable<i32>
}

hw.module @reg_with_instance_initial(in %clk: !seq.clock, in %in: i32, out out: i32) {
  %init = hw.instance "foo" @init_emitter () -> (out: !seq.immutable<i32>)
  // expected-error @below {{registers with initial values not directly defined by a seq.initial op not yet supported}}
  %1 = seq.compreg %in, %clk initial %init : i32
  hw.output %1 : i32
}

// -----
hw.module @firreg_with_async_reset(in %clk: !seq.clock, in %rst: i1, in %in: i32, out out: i32) {
  %c0_i32 = hw.constant 0 : i32
  // expected-error @below {{seq.firreg with async reset not yet supported}}
  %1 = seq.firreg %in clock %clk reset async %rst, %c0_i32 : i32
  hw.output %1 : i32
}