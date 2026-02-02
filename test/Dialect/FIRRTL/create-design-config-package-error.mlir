// RUN: circt-opt --firrtl-create-design-config-package %s -verify-diagnostics

firrtl.circuit "TestHarness" {
  firrtl.module @TestHarness(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %harness_param: !firrtl.uint<8>) attributes {convention = #firrtl<convention scalarized>} {
    firrtl.instance dut sym @sym @DUT()
    // expected-error @below {{design configuration 'harness_config' can only be used within the effective DUT}}
    %harness_config = firrtl.int.read_design_config_int "harness_config", 99, "" : !firrtl.uint<8>
    firrtl.matchingconnect %harness_param, %harness_config : !firrtl.uint<8>
  }
  firrtl.module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}], convention = #firrtl<convention scalarized>} {
  }
}
