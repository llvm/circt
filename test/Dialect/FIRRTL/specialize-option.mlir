// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-specialize-option{select=Platform=FPGA,Performance=Fast}))' %s | FileCheck %s


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

// CHECK-LABEL: firrtl.module @Foo
firrtl.module @Foo() {
  // CHECK: firrtl.instance inst_fpga @FPGATarget()
  firrtl.instance_choice inst_fpga @DefaultTarget alternatives @Platform
    { @FPGA -> @FPGATarget, @ASIC -> @ASICTarget } ()

  // CHECK: firrtl.instance inst_default @DefaultTarget()
  firrtl.instance_choice inst_default @DefaultTarget alternatives @Platform
    { @ASIC -> @ASICTarget } ()

  // CHECK: firrtl.instance inst_perf @FastTarget()
  firrtl.instance_choice inst_perf @DefaultTarget alternatives @Performance
    { @Fast -> @FastTarget } ()
}

}
