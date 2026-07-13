// RUN: circt-opt %s --lower-contracts | circt-bmc - -b 1 --module Caller_CheckContract_0 --emit-mlir -o - | FileCheck %s

// CHECK-NOT:   verif.symbolic_value
// CHECK-LABEL: func.func @Callee(
// CHECK-SAME:  %[[CALLEE_INPUT:.+]]: !smt.bv<8>, %[[CALLEE_RESULT:.+]]: !smt.bv<8>
// CHECK:       %[[CALLEE_EQ:.+]] = smt.eq %[[CALLEE_RESULT]], %[[CALLEE_INPUT]] : !smt.bv<8>
// CHECK:       %[[CALLEE_ITE:.+]] = smt.ite %[[CALLEE_EQ]],
// CHECK:       %[[CALLEE_ASSUME:.+]] = smt.eq %[[CALLEE_ITE]],
// CHECK:       smt.assert %[[CALLEE_ASSUME]]
hw.module @Callee(in %a : i8, out y : i8) {
  %0 = verif.contract %a : i8 {
    %1 = comb.icmp eq %0, %a : i8
    verif.ensure %1 : i1
  }
  hw.output %0 : i8
}

// CHECK-NOT:   func.func @Caller(
// CHECK-LABEL: func.func @Caller_CheckContract_0()
// CHECK:       smt.solver
hw.module @Caller(in %a : i8, out y : i8) {
  %0 = hw.instance "callee" @Callee(a: %a: i8) -> (y: i8)
  %1 = verif.contract %0 : i8 {
    %2 = comb.icmp eq %1, %a : i8
    verif.ensure %2 : i1
  }
  hw.output %1 : i8
}

// CHECK-LABEL: func.func @bmc_circuit(
// CHECK-SAME:  %[[INPUT:.+]]: !smt.bv<8>, %[[RESULT:.+]]: !smt.bv<8>)
// CHECK:       %[[ASSUME_EQ:.+]] = smt.eq %[[RESULT]], %[[INPUT]] : !smt.bv<8>
// CHECK:       %[[ASSUME_ITE:.+]] = smt.ite %[[ASSUME_EQ]],
// CHECK:       %[[ASSUME:.+]] = smt.eq %[[ASSUME_ITE]],
// CHECK:       smt.assert %[[ASSUME]]
// CHECK:       %[[ASSERT_EQ:.+]] = smt.eq %[[RESULT]], %[[INPUT]] : !smt.bv<8>
// CHECK:       %[[ASSERT_ITE:.+]] = smt.ite %[[ASSERT_EQ]],
// CHECK:       %[[ASSERT:.+]] = smt.eq %[[ASSERT_ITE]],
// CHECK:       %[[VIOLATED:.+]] = smt.not %[[ASSERT]]
// CHECK:       smt.assert %[[VIOLATED]]
// CHECK-NOT:   verif.symbolic_value
