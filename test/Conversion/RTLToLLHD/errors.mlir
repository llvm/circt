// RUN: circt-opt -convert-rtl-to-llhd -split-input-file -verify-diagnostics %s

module {
  // Since HW-to-LLHD needs to construct a zero value for temporary signals,
  // we don't support non-IntegerType arguments to instances.
  rtl.module @sub(%in: f16) -> (%out: f16) {
    rtl.output %in: f16
  }
  rtl.module @test(%in: f16) -> (%out: f16) {
    // expected-error @+1 {{failed to legalize operation 'rtl.instance'}}
    %0 = rtl.instance "sub1" @sub (%in) : (f16) -> (f16)
    rtl.output %0: f16
  }
}
