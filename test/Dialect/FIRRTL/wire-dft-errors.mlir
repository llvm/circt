// RUN: circt-opt -firrtl-dft -verify-diagnostics -split-input-file %s

// expected-error @+1 {{no DUT module found}}
firrtl.circuit "NoDuts" {
  firrtl.module @NoDuts() {
    %test_en1 = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]}: !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "TwoDuts" {
  firrtl.module @TwoDuts() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}
  // expected-error @+2 {{more than one module marked DUT}}
  // expected-note  @-2 {{first module here}}
  firrtl.module @TwoDuts0() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}
}

// -----

firrtl.circuit "TwoSignals" {
  firrtl.module @TwoSignals(in %test_en0: !firrtl.uint<1>) attributes {portAnnotations = [[{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]]} {
    // expected-error @+2 {{more than one thing marked as a DFT enable}}
    // expected-note  @-2 {{first thing defined here}}
    %test_en1 = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]}: !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "TwoEnables" {

  firrtl.extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  firrtl.module @TestEn() {
    // expected-error @+1 {{mutliple instantiations of the DFT enable signal}}
    %test_en = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
  
  firrtl.module @TwoEnables() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // expected-note @+1 {{second instance here}}
    firrtl.instance test_en0 @TestEn()
    // expected-note @+1 {{first instance here}}
    firrtl.instance test_en1 @TestEn()
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = firrtl.instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
  }
}
