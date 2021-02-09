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

// -----

module {
  // CHECK-LABEL: llhd.entity @sub
  // CHECK-SAME: ({{.+}} : !llhd.sig<i1>) ->
  // CHECK-SAME: ({{.+}} : !llhd.sig<i1>)
  rtl.module @sub(%in: i1) -> (%out: i1) {
    rtl.output %in : i1
  }

  // CHECK-LABEL: llhd.entity @top
  // CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) ->
  // CHECK-SAME: (%[[OUT:.+]] : !llhd.sig<i1>)
  rtl.module @top(%in: i1) -> (%out: i1) {
    // CHECK: llhd.inst "sub1" @sub(%[[IN]]) -> (%[[OUT]]) : (!llhd.sig<i1>) -> !llhd.sig<i1>
    %0 = rtl.instance "sub1" @sub (%in) : (i1) -> i1
    rtl.output %0 : i1
  }
}
