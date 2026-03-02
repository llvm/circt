// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-instance-choice))' %s --verify-diagnostics

// Test case demonstrating valid instance choice scenarios (no nesting)
firrtl.circuit "Top" {
  firrtl.option @Optimization {
    firrtl.option_case @Speed
  }

  firrtl.module private @Leaf() { }

  firrtl.module private @Middle() {
    firrtl.instance_choice leaf @Leaf alternatives @Optimization { @Speed -> @Leaf } ()
  }

  firrtl.module public @Top() {
    firrtl.instance m @Middle()
  }
}
