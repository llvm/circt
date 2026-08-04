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

func.func @duplicate_bmc_trace_names(%step: i32, %a: !smt.bv<8>, %b: !smt.bv<8>) {
  // expected-note @+1 {{first BMC trace with this name is here}}
  verif.bmc.trace %step, "x", %a : i32, !smt.bv<8>
  // expected-error @+1 {{duplicate BMC trace name 'x'}}
  verif.bmc.trace %step, "x", %b : i32, !smt.bv<8>
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

// -----

// Make sure we don't delete dbg.scope ops whose users aren't being erased
func.func @multiple_set_logics() {
  // expected-error @below {{failed to legalize operation 'dbg.scope'}}
  %scope = dbg.scope "a", "A"
  builtin.unrealized_conversion_cast %scope : !dbg.scope to i32
  func.return
}
