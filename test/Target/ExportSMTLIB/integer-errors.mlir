// RUN: circt-translate --export-smtlib %s --split-input-file --verify-diagnostics

smt.solver () : () -> () {
  %0 = smt.int.constant 5
  // expected-error @below {{int2bv operations are not supported for SMTLIB emission}}
  %1 = smt.int2bv %0 width 4 : !smt.bv<4>
}
