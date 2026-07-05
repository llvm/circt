// RUN: circt-opt %s --lower-contracts | circt-bmc - -b 1 --module Foo_CheckContract_0 --emit-mlir -o - | FileCheck %s

// CHECK-NOT:   func.func @Foo(
// CHECK-LABEL: func.func @Foo_CheckContract_0()
// CHECK:       smt.solver
// CHECK:       llvm.call @printf
// CHECK-NOT:   func.func @Foo(
hw.module @Foo(in %a : i1, in %b : i1, out z : i1) {
  %0 = comb.xor %a, %b : i1
  %1 = verif.contract %0 : i1 {
    %2 = comb.add %a, %b : i1
    %3 = comb.icmp eq %1, %2 : i1
    verif.ensure %3 : i1
  }
  hw.output %1 : i1
}
