// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-specialize-option{select=Platform=SuperDuperSystem,Bad=Dummy,worst}))' --verify-diagnostics %s

// expected-warning @below {{invalid option case "SuperDuperSystem"}}
// expected-warning @below {{unknown option "Bad"}}
// expected-error @below {{invalid option format: "worst"}}
firrtl.circuit "Foo" {

firrtl.option @Platform {
  firrtl.option_case @FPGA
  firrtl.option_case @ASIC
}

firrtl.option @Performance {
  firrtl.option_case @Fast
  firrtl.option_case @Small
}

}
