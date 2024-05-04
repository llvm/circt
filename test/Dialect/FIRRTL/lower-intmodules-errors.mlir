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
