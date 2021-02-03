// RUN: circt-opt -convert-rtl-to-llhd -split-input-file -verify-diagnostics %s | FileCheck %s

module {
  // CHECK-LABEL: llhd.entity @test
  // CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) ->
  // CHECK-SAME: (%[[OUT:.+]] : !llhd.sig<i1>)
  rtl.module @test(%in: i1) -> (%out: i1) {
    // CHECK: llhd.con %[[OUT]], %[[IN]]
    rtl.output %in: i1
  }
}
