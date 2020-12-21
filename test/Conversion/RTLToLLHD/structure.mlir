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
