// RUN: circt-opt -convert-rtl-to-llhd -split-input-file -verify-diagnostics %s | FileCheck %s

module {
  // CHECK-LABEL: @test
  // expected-remark @+1 {{TODO}}
  rtl.module @test(%in: i1) -> (%out: i1) {
    rtl.output %in: i1
  }
}
