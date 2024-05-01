// RUN: circt-translate --export-smtlib %s > %t && z3 %t 2>&1 | FileCheck %s
// RUN: circt-translate --export-smtlib --smtlibexport-inline-single-use-values %s > %t && z3 %t 2>&1 | FileCheck %s
// REQUIRES: z3

// Quantifiers Attributes
smt.solver () : () -> () {

  %true = smt.constant true

  %7 = smt.exists weight 2 {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    %5 = smt.implies %4, %true
    smt.yield %5 : !smt.bool
  } patterns {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %4: !smt.bool
  }
  smt.assert %7

  smt.check sat {} unknown {} unsat {}
}

// CHECK-NOT: WARNING
// CHECK-NOT: warning
// CHECK-NOT: ERROR
// CHECK-NOT: error
// CHECK-NOT: unsat
// CHECK: sat
