// RUN: circt-opt -convert-rtl-to-llhd -split-input-file -verify-diagnostics %s

module {
  // expected-error @+1 {{failed to legalize operation 'rtl.module'}}
  rtl.module @test(%in: f16) -> (%out: f16) {
    rtl.output %in: f16
  }
}
