// RUN: circt-translate --export-smtlib %s | FileCheck %s
// RUN: circt-translate --export-smtlib --smtlibexport-inline-single-use-values %s | FileCheck %s --check-prefix=CHECK-INLINED

smt.solver () : () -> () {

  %true = smt.constant true

  %6 = smt.exists {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    %5 = smt.implies %4, %true
    smt.yield %5 : !smt.bool
  } patterns {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %4: !smt.bool
  }
  smt.assert %6

  %7 = smt.exists weight 2 {
    ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    %5 = smt.implies %4, %true
    smt.yield %5 : !smt.bool
  }
  smt.assert %7
}