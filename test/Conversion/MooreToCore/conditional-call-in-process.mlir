// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

func.func private @lhs() -> !moore.i32
func.func private @rhs() -> !moore.i32

// Regression for conditionals with call-bearing branches inside procedures.
// These must lower to CFG branches without leaving orphaned scf.yield ops.
// CHECK-LABEL: hw.module @ConditionalCallInInitial
moore.module @ConditionalCallInInitial() {
  // CHECK: llhd.process {
  // CHECK:   [[COND:%.+]] = hw.constant true
  // CHECK:   cf.cond_br [[COND]], ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
  // CHECK: ^[[THEN]]:
  // CHECK:   [[LHS:%.+]] = func.call @lhs() : () -> i32
  // CHECK:   cf.br ^[[JOIN:bb[0-9]+]]([[LHS]] : i32)
  // CHECK: ^[[ELSE]]:
  // CHECK:   [[RHS:%.+]] = func.call @rhs() : () -> i32
  // CHECK:   cf.br ^[[JOIN]]([[RHS]] : i32)
  // CHECK: ^[[JOIN]]([[VALUE:%.+]]: i32):
  // CHECK:   cf.br ^[[EXIT:bb[0-9]+]]
  // CHECK: ^[[EXIT]]:
  // CHECK:   dbg.variable "v", [[VALUE]] : i32
  moore.procedure initial {
    %cond = moore.constant 1 : i1
    %0 = moore.conditional %cond : i1 -> i32 {
      %1 = func.call @lhs() : () -> !moore.i32
      moore.yield %1 : i32
    } {
      %2 = func.call @rhs() : () -> !moore.i32
      moore.yield %2 : i32
    }
    dbg.variable "v", %0 : !moore.i32
    moore.return
  }
}
