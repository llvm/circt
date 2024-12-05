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
