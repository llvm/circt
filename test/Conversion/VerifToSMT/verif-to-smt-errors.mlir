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
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [unit]
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
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [unit]
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
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [unit]
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
  %bmc = verif.bmc bound 10 num_regs 1 initial_values [unit]
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
