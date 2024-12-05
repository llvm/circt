// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-specialize-option{select=Platform=FPGA}))' --verify-diagnostics %s

firrtl.circuit "Foo" {
firrtl.option @Platform {
  firrtl.option_case @FPGA
  firrtl.option_case @ASIC
}
firrtl.option @Performance {
  firrtl.option_case @Fast
  firrtl.option_case @Small
}

firrtl.module private @DefaultTarget() {}
firrtl.module private @FPGATarget() {}
firrtl.module private @ASICTarget() {}
firrtl.module private @FastTarget() {}

firrtl.module @Foo() {
  firrtl.instance_choice inst_fpga @DefaultTarget alternatives @Platform
    { @FPGA -> @FPGATarget, @ASIC -> @ASICTarget } ()

  firrtl.instance_choice inst_default @DefaultTarget alternatives @Platform
    { @ASIC -> @ASICTarget } ()

  // expected-error @below {{missing specialization for option "Performance"}}
  firrtl.instance_choice inst_perf @DefaultTarget alternatives @Performance
    { @Fast -> @FastTarget } ()
}

}
