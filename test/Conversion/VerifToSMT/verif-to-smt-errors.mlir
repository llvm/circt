// RUN: circt-opt %s --convert-verif-to-smt --split-input-file --verify-diagnostics

func.func @assert_with_unsupported_property_type(%arg0: !smt.bv<1>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to !ltl.sequence
  // expected-error @below {{failed to legalize operation 'verif.assert' that was explicitly marked illegal}}
  verif.assert %0 : !ltl.sequence
  return
}

// -----

func.func @assert_with_unsupported_property_type(%arg0: !smt.bv<1>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !smt.bv<1> to !ltl.property
  // expected-error @below {{failed to legalize operation 'verif.assert' that was explicitly marked illegal}}
  verif.assert %0 : !ltl.property
  return
}

// -----

func.func @multiple_assertions_bmc() -> (i1) {
  // expected-error @below {{bounded model checking problems with multiple assertions are not yet correctly handled - instead, you can assert the conjunction of your assertions}}
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32, %arg1: i32):
    %c1_i32 = hw.constant 1 : i32
    %cond1 = comb.icmp ugt %arg0, %c1_i32 : i32
    verif.assert %cond1 : i1
    %cond2 = comb.icmp ugt %arg1, %c1_i32 : i32
    verif.assert %cond2 : i1
    %sum = comb.add %arg0, %arg1 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

// -----

func.func @multiple_asserting_modules_bmc() -> (i1) {
  // expected-error @below {{bounded model checking problems with multiple assertions are not yet correctly handled - instead, you can assert the conjunction of your assertions}}
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32, %arg1: i1, %arg2: i1):
    hw.instance "" @OneAssertion(x: %arg1: i1) -> ()
    hw.instance "" @OneAssertion(x: %arg2: i1) -> ()
    %sum = comb.add %arg0, %arg0 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

hw.module @OneAssertion(in %x: i1) {
  verif.assert %x : i1
}

// -----

func.func @two_separated_assertions() -> (i1) {
  // expected-error @below {{bounded model checking problems with multiple assertions are not yet correctly handled - instead, you can assert the conjunction of your assertions}}
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32, %arg1: i1, %arg2: i1):
    hw.instance "" @OneAssertion(x: %arg1: i1) -> ()
    verif.assert %arg2 : i1
    %sum = comb.add %arg0, %arg0 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

hw.module @OneAssertion(in %x: i1) {
  verif.assert %x : i1
}

// -----

func.func @multiple_nested_assertions() -> (i1) {
  // expected-error @below {{bounded model checking problems with multiple assertions are not yet correctly handled - instead, you can assert the conjunction of your assertions}}
  %bmc = verif.bmc bound 10 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%arg0: i32, %arg1: i1, %arg2: i1):
    hw.instance "" @TwoAssertions(x: %arg1: i1, y: %arg2: i1) -> ()
    %sum = comb.add %arg0, %arg0 : i32
    verif.yield %sum : i32
  }
  func.return %bmc : i1
}

hw.module @TwoAssertions(in %x: i1, in %y: i1) {
  verif.assert %x : i1
  verif.assert %y : i1
}

// -----

func.func @multiple_clocks() -> (i1) {
  // expected-error @below {{only modules with one or zero clocks are currently supported}}
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [unit]
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk, %clk : !seq.clock, !seq.clock
  }
  loop {
    ^bb0(%clock0: !seq.clock, %clock1: !seq.clock):
    verif.yield %clock0, %clock1 : !seq.clock, !seq.clock
  }
  circuit {
  ^bb0(%clock0: !seq.clock, %clock1: !seq.clock, %arg0: i32):
    %c1_i32 = hw.constant 1 : i32
    %cond1 = comb.icmp ugt %arg0, %c1_i32 : i32
    verif.assert %cond1 : i1
    verif.yield %arg0 : i32
  }
  func.return %bmc : i1
}

// -----

func.func @multiple_clocks() -> (i1) {
  // expected-error @below {{initial values are currently only supported for registers with integer types}}
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [0]
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk : !seq.clock
  }
  loop {
    ^bb0(%clk: !seq.clock):
    verif.yield %clk: !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: !hw.array<2xi32>):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %arg0 : !hw.array<2xi32>
  }
  func.return %bmc : i1
}

// -----

func.func @wrong_initial_type() -> (i1) {
  // expected-error @below {{type of initial value does not match type of initialized register}}
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [-1 : i7]
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk : !seq.clock
  }
  loop {
    ^bb0(%clk: !seq.clock):
    verif.yield %clk: !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %arg0: i8):
    %true = hw.constant true
    verif.assert %true : i1
    verif.yield %arg0 : i8
  }
  func.return %bmc : i1
}

// -----

func.func @refines_non_primitive_free_var() -> () {
  // expected-error @below {{failed to legalize operation 'verif.refines' that was explicitly marked illegal}}
  verif.refines first {
  ^bb0(%arg0: !smt.bv<4>):
    // expected-error @below {{Uninterpreted function of non-primitive type cannot be converted.}}
    %nondetar = smt.declare_fun : !smt.array<[!smt.bv<4> -> !smt.bv<32>]>
    %sel = smt.array.select %nondetar[%arg0] : !smt.array<[!smt.bv<4> -> !smt.bv<32>]>
    %cc = builtin.unrealized_conversion_cast %sel : !smt.bv<32> to i32
    verif.yield %cc : i32
  } second {
  ^bb0(%arg0: !smt.bv<4>):
    %const = smt.bv.constant #smt.bv<0> : !smt.bv<32>
    %cc = builtin.unrealized_conversion_cast %const : !smt.bv<32> to i32
    verif.yield %cc : i32
  }
  return
}
