// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test_bmc() -> i1
// CHECK-NEXT: [[TRUE:%.+]] = arith.constant true
// CHECK-NEXT: return [[TRUE]] : i1

func.func @test_bmc() -> (i1) {
  %bmc = verif.bmc bound 10 num_regs 0 initial_values [] attributes {ignore_asserts_until = 3 : i64}
  init {
  }
  loop {
  }
  circuit {
  ^bb0(%arg0: i32):
    verif.yield %arg0 : i32
  }
  func.return %bmc : i1
}
