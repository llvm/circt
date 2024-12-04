// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-specialize-option))' --verify-diagnostics %s

// expected-warning @below {{invalid option case "SuperDuperSystem"}}
// expected-warning @below {{unknown option "Bad"}}
firrtl.circuit "Foo"  attributes {
  select_inst_choice = ["Platform=SuperDuperSystem" ,"Bad=Dummy,worst"]
}
{

firrtl.extmodule @Foo ()

firrtl.option @Platform {
  firrtl.option_case @FPGA
  firrtl.option_case @ASIC
}

firrtl.option @Performance {
  firrtl.option_case @Fast
  firrtl.option_case @Small
}

}
