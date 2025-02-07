// RUN: circt-opt %s --lower-smt-to-z3-llvm --split-input-file --verify-diagnostics

func.func @multiple_set_logics() {
  // expected-error @below {{multiple set-logic operations found in one solver operation - Z3 only supports setting the logic once}}
  smt.solver () : () -> () {
    smt.set_logic "HORN"
    smt.set_logic "AUFLIA"
    smt.yield
  }
  func.return
}

// -----

func.func @multiple_set_logics() {
  // expected-error @below {{set-logic operation must be the first non-constant operation in a solver operation}}
  smt.solver () : () -> () {
    smt.check sat {} unknown {} unsat {}
    smt.set_logic "HORN"
    smt.yield
  }
  func.return
}
