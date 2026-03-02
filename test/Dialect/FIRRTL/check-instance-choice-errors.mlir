// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-instance-choice))' %s --verify-diagnostics --split-input-file

// Test case: Nested instance choices are not allowed
firrtl.circuit "TestNested" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
    firrtl.option_case @ASIC
  }
  firrtl.option @Optimization {
    firrtl.option_case @Speed
    firrtl.option_case @Area
  }
  firrtl.module private @Level3() {}
  firrtl.module private @Level2() {
    // expected-error @+1 {{instance choice within another instance choice is not allowed}}
    firrtl.instance_choice level3 @Level3 alternatives @Optimization { @Speed -> @Level3, @Area -> @Level3 } ()
  }
  firrtl.module private @Level1() {
    // expected-note @+1 {{is the parent instance choice}}
    firrtl.instance_choice level2 @Level2 alternatives @Platform { @FPGA -> @Level2, @ASIC -> @Level2 } ()
  }
  firrtl.module @TestNested() {
    firrtl.instance level1 @Level1()
  }
}
