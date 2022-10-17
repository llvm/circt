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


// -----

// Test enable signal that isn't reachable from DUT.
// expected-error @below {{unable to connect enable signal and DUT, may not be reachable from top-level module}}
firrtl.circuit "EnableNotReachable" {
  firrtl.extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  firrtl.module @TestEn() {
    // expected-note @below {{enable signal here}}
    %test_en = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }
  // expected-note @below {{DUT here}}
  // expected-note @below {{top-level module here}}
  firrtl.module @EnableNotReachable() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %eicg_in, %eicg_test_en, %eicg_en, %eicg_out = firrtl.instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
  }
}

// -----

// Test clock gate instantiated both in and outside DUT
firrtl.circuit "InAndOutOfDUT" {
  firrtl.extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}

  // expected-error @below {{clock gates within DUT must not be instantiated outside the DUT}}
  firrtl.module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    firrtl.instance a @A()
    %test_en = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation"}]} : !firrtl.uint<1>
  }

  firrtl.module @A() {
    %in, %test_en, %en, %out = firrtl.instance eicg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
  }

  firrtl.module @InAndOutOfDUT() {
    firrtl.instance a @A()
    firrtl.instance d @DUT()
  }
}
