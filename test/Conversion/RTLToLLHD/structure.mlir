// RUN: circt-opt -convert-rtl-to-llhd -split-input-file -verify-diagnostics %s | FileCheck %s

module {
  // CHECK-LABEL: llhd.entity @test
  // CHECK-SAME: (%[[IN:.+]] : !llhd.sig<i1>) ->
  // CHECK-SAME: (%[[OUT:.+]] : !llhd.sig<i1>)
  rtl.module @test(%in: i1) -> (%out: i1) {
    // CHECK: %[[DELTA:.*]] = llhd.const #llhd.time<0ns, 1d, 0e>
    // CHECK: %[[PRB:.*]] = llhd.prb %[[IN]]
    // CHECK: llhd.drv %[[OUT]], %[[PRB]] after %[[DELTA]]
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
    // CHECK: %[[TMP_SIGNAL:.+]] = llhd.sig "{{.+}}" {{.+}} : i1
    // CHECK: llhd.inst "sub1" @sub(%[[IN]]) -> (%[[TMP_SIGNAL]]) : (!llhd.sig<i1>) -> !llhd.sig<i1>
    %0 = rtl.instance "sub1" @sub (%in) : (i1) -> i1
    // CHECK: %[[TMP_PROBED:.+]] = llhd.prb %[[TMP_SIGNAL]]
    // CHECK: llhd.drv %[[OUT]], %[[TMP_PROBED]]
    rtl.output %0 : i1
  }
}
