// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intmodules{fixup-eicg-wrapper}))' --split-input-file --verify-diagnostics %s

firrtl.circuit "EICGWithInstAnno" {
  firrtl.extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper"}
  hw.hierpath private @nla [@EICGWithInstAnno::@ckg, @EICG_wrapper]
  firrtl.module @EICGWithInstAnno(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // expected-error @below {{EICG_wrapper instance cannot have annotations since it is an intrinsic}}
    firrtl.instance ckg sym @ckg {annotations = [{circt.nonlocal = @nla, class = "DummyA"}]} @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
  }
}

// -----

firrtl.circuit "EICGWithModuleAnno" {
  // expected-error @below {{EICG_wrapper cannot have annotations since it is an intrinsic}}
  firrtl.extmodule @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {defname = "EICG_wrapper", annotations = [{class = "DummyA"}]}
  firrtl.module @EICGWithModuleAnno(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>) {
    firrtl.instance ckg @EICG_wrapper(in in: !firrtl.clock, in test_en: !firrtl.uint<1>, in en: !firrtl.uint<1>, out out: !firrtl.clock)
  }
}

// -----

firrtl.circuit "BasicIntmoduleInstances" {

  firrtl.option @Opt {
      firrtl.option_case @A
  }

  firrtl.intmodule @test(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {intrinsic = "circt.sizeof"}

  firrtl.module @BasicIntmoduleInstances() {
    // expected-error @below {{intmodule must be instantiated with instance op, not via 'firrtl.instance_choice'}}
    %i1, %size = firrtl.instance_choice inst interesting_name @test alternatives @Opt { @A -> @test }(in i : !firrtl.clock, out size : !firrtl.uint<32>)
  }
}
