// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-instance-choice))' %s --verify-diagnostics --split-input-file

// Test case: Triple nested with different options - should ERROR at first conflict
firrtl.circuit "TestTripleNested" {
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
    // expected-error @+1 {{nested instance choice with option 'Optimization' conflicts with option 'Platform' already on the path from public module 'TestTripleNested'}}
    firrtl.instance_choice level3 @Level3 alternatives @Optimization { @Speed -> @Level3, @Area -> @Level3 } ()
  }
  firrtl.module private @Level1() {
    firrtl.instance_choice level2 @Level2 alternatives @Platform { @FPGA -> @Level2, @ASIC -> @Level2 } ()
  }
  // expected-note @+1 {{public module here}}
  firrtl.module @TestTripleNested() {
    firrtl.instance level1 @Level1()
  }
}

// -----

firrtl.circuit "TestSameOptionDifferentCase" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
    firrtl.option_case @ASIC
  }
  firrtl.module private @Leaf() {}
  firrtl.module private @Dummy() {}
  firrtl.module private @Inner() {
    // When reached via FPGA case from Outer, the default case here conflicts
    // expected-error @+1 {{'firrtl.instance_choice' op nested instance choice with option 'Platform' and default case conflicts with case '@Platform::@FPGA' already on the path from public module 'TestSameOptionDifferentCase'}}
    firrtl.instance_choice leaf @Dummy alternatives @Platform { @ASIC -> @Leaf } ()
  }
  firrtl.module private @Outer() {
    // This uses FPGA case to reach Inner
    firrtl.instance_choice inner @Dummy alternatives @Platform { @FPGA -> @Inner } ()
  }
  // expected-note @+1 {{public module here}}
  firrtl.module @TestSameOptionDifferentCase() {
    firrtl.instance outer @Outer()
  }
}

