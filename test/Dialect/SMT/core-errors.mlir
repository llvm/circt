// RUN: circt-opt %s --split-input-file --verify-diagnostics

func.func @solver_isolated_from_above(%arg0: !smt.bool) {
  // expected-note @below {{required by region isolation constraints}}
  smt.solver() : () -> () {
    // expected-error @below {{using value defined outside the region}}
    smt.assert %arg0
  }
  return
}

// -----

func.func @no_smt_value_enters_solver(%arg0: !smt.bool) {
  // expected-error @below {{operand #0 must be variadic of any non-smt type, but got '!smt.bool'}}
  smt.solver(%arg0) : (!smt.bool) -> () {
  ^bb0(%arg1: !smt.bool):
    smt.assert %arg1
  }
  return
}

// -----

func.func @no_smt_value_exits_solver() {
  // expected-error @below {{result #0 must be variadic of any non-smt type, but got '!smt.bool'}}
  %0 = smt.solver() : () -> !smt.bool {
    %a = smt.declare_const "a" : !smt.bool
    smt.yield %a : !smt.bool
  }
  return
}

// -----

func.func @block_args_and_inputs_match() {
  // expected-error @below {{block argument types must match the types of the 'inputs'}}
  smt.solver() : () -> () {
    ^bb0(%arg0: i32):
  }
  return
}

// -----

func.func @solver_yield_operands_and_results_match() {
  // expected-error @below {{types of yielded values must match return values}}
  smt.solver() : () -> () {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  }
  return
}

// -----

func.func @check_yield_operands_and_results_match() {
  // expected-error @below {{types of yielded values in 'unsat' region must match return values}}
  %0 = smt.check sat {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } unknown {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } unsat { } -> i32
  return
}

// -----

func.func @check_yield_operands_and_results_match() {
  // expected-error @below {{types of yielded values in 'unknown' region must match return values}}
  %0 = smt.check sat {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } unknown {
  } unsat {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } -> i32
  return
}

// -----

func.func @check_yield_operands_and_results_match() {
  // expected-error @below {{types of yielded values in 'sat' region must match return values}}
  %0 = smt.check sat {
  } unknown {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } unsat {
    %1 = hw.constant 0 : i32
    smt.yield %1 : i32
  } -> i32
  return
}

// -----

func.func @check_no_block_arguments() {
  // expected-error @below {{region #0 should have no arguments}}
  smt.check sat {
  ^bb0(%arg0: i32):
  } unknown {
  } unsat {
  }
  return
}

// -----

func.func @check_no_block_arguments() {
  // expected-error @below {{region #1 should have no arguments}}
  smt.check sat {
  } unknown {
  ^bb0(%arg0: i32):
  } unsat {
  }
  return
}

// -----

func.func @check_no_block_arguments() {
  // expected-error @below {{region #2 should have no arguments}}
  smt.check sat {
  } unknown {
  } unsat {
  ^bb0(%arg0: i32):
  }
  return
}
