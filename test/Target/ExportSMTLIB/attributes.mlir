// RUN: circt-translate --export-smtlib %s | FileCheck %s
// RUN: circt-translate --export-smtlib --smtlibexport-inline-single-use-values %s | FileCheck %s --check-prefix=CHECK-INLINED

smt.solver () : () -> () {


  // CHECK: (assert (let (([[V10:.+]] (forall (([[A:.+]] Int) ([[B:.+]] Int))
  // CHECK:                           (let (([[V11:.+]] (= [[A]] [[B]])))
  // CHECK:                           [[V11]]))))
  // CHECK:         [[V10]]))

  // CHECK-INLINED: (assert (forall (([[A:.+]] Int) ([[B:.+]] Int))
  // CHECK-INLINED:                 (= [[A]] [[B]])))
  %1 = smt.forall ["a", "b"] {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %2 : !smt.bool
  }
  smt.assert %1

  // CHECK: (assert (let (([[V12:.+]] (exists (([[V13:.+]] Int) ([[V14:.+]] Int))
  // CHECK:                           (let (([[V15:.+]] (= [[V13]] [[V14]])))
  // CHECK:                           ( ! [[V15]] 2)))))
  // CHECK:         [[V12]]))

  // CHECK-INLINED: (assert (exists (([[A:.+]] Int) ([[B:.+]] Int))
  // CHECK-INLINED:                ( ! (= [[A]] [[B]]) 2)))
  %2 = smt.exists weight 2 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %3 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %3 : !smt.bool
  }
  smt.assert %2

  // CHECK: (reset)
  // CHECK-INLINED: (reset)
}
