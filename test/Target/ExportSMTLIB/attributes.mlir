// RUN: circt-translate --export-smtlib %s | FileCheck %s
// RUN: circt-translate --export-smtlib --smtlibexport-inline-single-use-values %s | FileCheck %s --check-prefix=CHECK-INLINED

smt.solver () : () -> () {

  %true = smt.constant true


  // CHECK: (assert (let (([[V10:.+]] (forall (([[A:.+]] Int) ([[B:.+]] Int))
  // CHECK:                           ( ! (let (([[V11:.+]] (= [[A]] [[B]])))
  // CHECK:                           [[V11]]) :weight 2))))
  // CHECK:         [[V10]]))

  // CHECK-INLINED: (assert (forall (([[A:.+]] Int) ([[B:.+]] Int))
  // CHECK-INLINED:                 ( ! (= [[A]] [[B]]) :weight 2)))
  %1 = smt.forall ["a", "b"] weight 2 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %2 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %2 : !smt.bool
  }
  smt.assert %1

  // CHECK: (assert (let (([[V12:.+]] (exists (([[V13:.+]] Int) ([[V14:.+]] Int))
  // CHECK:                           ( ! (let (([[V15:.+]] (= [[V13]] [[V14]])))
  // CHECK:                           [[V15]]) :weight 2))))
  // CHECK:         [[V12]]))

  // CHECK-INLINED: (assert (exists (([[A:.+]] Int) ([[B:.+]] Int))
  // CHECK-INLINED:                 ( ! (= [[A]] [[B]]) :weight 2)))
  %2 = smt.exists weight 2 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %3 = smt.eq %arg2, %arg3 : !smt.int
    smt.yield %3 : !smt.bool
  }
  smt.assert %2


  // CHECK: (assert (let (([[V16:.+]] (exists (([[V17:.+]] Int) ([[V18:.+]] Int))
  // CHECK:                           ( ! (let (([[V19:.+]] (= [[V17]] [[V18]])))
  // CHECK:                           (let (([[V20:.+]] (=> [[V19:.+]] true)))
  // CHECK:                           [[V20:.+]])) :weight 2))))
  // CHECK:         [[V16]]))

  %3 = smt.exists weight 2 {
  ^bb0(%arg2: !smt.int, %arg3: !smt.int):
    %4 = smt.eq %arg2, %arg3 : !smt.int
    %5 = smt.implies %4, %true
    smt.yield %5 : !smt.bool
  }
  smt.assert %3


  // CHECK: (assert (let (([[V21:.+]] (exists (([[V22:.+]] Int) ([[V23:.+]] Int))
  // CHECK:                           ( ! (let (([[V24:.+]] (= [[V22]] [[V23]])))
  // CHECK:                           (let (([[V25:.+]] (=> [[V24:.+]] true)))
  // CHECK:                           [[V25:.+]])) 
  // CHECK                            :pattern ( ( let (([[V26:.+]] (= [[V22:.+]] [[V23:.+]])))
  // CHECK:                   [[V26:.+]]))))))
  // CHECK:         [[V21]]))


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


  // CHECK: (assert (let (([[V27:.+]] (exists (([[V28:.+]] Int) ([[V29:.+]] Int))
  // CHECK:                           ( ! (let (([[V30:.+]] (= [[V28]] [[V29]])))
  // CHECK:                           (let (([[V31:.+]] (=> [[V30:.+]] true)))
  // CHECK:                           [[V31:.+]])) :weight 2
  // CHECK                            :pattern ( ( let (([[V32:.+]] (= [[V28:.+]] [[V29:.+]])))
  // CHECK:                   [[V32:.+]]))))))
  // CHECK:         [[V27]]))


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

  // CHECK: (reset)
  // CHECK-INLINED: (reset)
}
