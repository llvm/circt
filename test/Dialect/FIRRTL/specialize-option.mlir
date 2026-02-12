// RUN: circt-opt --firrtl-specialize-option %s | FileCheck %s --check-prefixes=CHECK,NOT-DEFAULT
// RUN: circt-opt --firrtl-specialize-option='select-default-for-unspecified-instance-choice=true' %s | FileCheck %s --check-prefixes=CHECK,DEFAULT



firrtl.circuit "Foo" attributes {
  select_inst_choice = ["Platform=FPGA" ,"Performance=Fast"]
}
{

// CHECK-NOT: firrtl.option @Platform 
firrtl.option @Platform {
  firrtl.option_case @FPGA
  firrtl.option_case @ASIC
}
// CHECK-NOT: firrtl.option @Performance 
firrtl.option @Performance {
  firrtl.option_case @Fast
  firrtl.option_case @Small
}

firrtl.option @NotSelected {
  firrtl.option_case @Fast
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

  // NOT-DEFAULT: firrtl.instance_choice inst_keep @DefaultTarget alternatives @NotSelected
  // DEFAULT: firrtl.instance inst_keep @DefaultTarget()
  firrtl.instance_choice inst_keep @DefaultTarget alternatives @NotSelected
    { @Fast -> @FastTarget } ()
}

}
