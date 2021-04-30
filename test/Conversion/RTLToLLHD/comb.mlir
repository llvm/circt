// RUN: circt-opt -convert-rtl-to-llhd -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: llhd.entity @CombOp
// CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) ->
// CHECK-SAME: (%[[OUT:.+]] : !llhd.sig<i1>)
rtl.module @CombOp(%in: i1) -> (%out: i1) {
  // CHECK: %[[PRB:.+]] = llhd.prb %[[IN]]
  // CHECK: %[[ADD:.+]] = comb.add %[[PRB]], %[[PRB]]
  // CHECK: %[[SIG:.+]] = llhd.sig {{.+}} %[[ADD]]
  // CHECK: llhd.con %[[OUT]], %[[SIG]]
  %0 = comb.add %in, %in : i1
  rtl.output %0: i1
}
