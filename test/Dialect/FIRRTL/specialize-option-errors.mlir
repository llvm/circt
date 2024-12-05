// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-specialize-option))' --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Incorrect option case 
//===----------------------------------------------------------------------===//

// expected-error @below {{invalid option case "SuperDuperSystem"}}
firrtl.circuit "Foo"  attributes {
  select_inst_choice = ["Platform=SuperDuperSystem"]
}
{

firrtl.extmodule @Foo ()

firrtl.option @Platform {
  firrtl.option_case @FPGA
  firrtl.option_case @ASIC
}
}

//===----------------------------------------------------------------------===//
// Incorrect option name 
//===----------------------------------------------------------------------===//

// expected-error @below {{unknown option "Patform"}}
firrtl.circuit "Foo"  attributes {
  select_inst_choice = ["Patform=ASIC"]
}
{

firrtl.extmodule @Foo ()

firrtl.option @Platform {
  firrtl.option_case @FPGA
  firrtl.option_case @ASIC
}
}

//===----------------------------------------------------------------------===//
//  Partially specified options 
//===----------------------------------------------------------------------===//


firrtl.circuit "Foo" attributes {
  select_inst_choice = ["Platform=FPGA"]
}
{
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
