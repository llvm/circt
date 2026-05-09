// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-populate-instance-choice-symbols))' %s --verify-diagnostics

// Test that conflicts with ABI-defined macro names are detected.
firrtl.circuit "ConflictTest" {
  // Pre-existing module with the same name as the ABI-defined macro.
  // This creates a conflict in the CircuitNamespace.
  firrtl.module @"targets$Platform$ASIC"() { }

  // expected-error @+2 {{case macro name conflicts with existing symbol 'targets$Platform$ASIC' (existing symbol is 'firrtl.module')}}
  firrtl.option @Platform {
    firrtl.option_case @ASIC
  }

  firrtl.module private @ModuleDefault() { }
  firrtl.module private @ModuleASIC() { }

  firrtl.module @ConflictTest() {
    firrtl.instance_choice inst @ModuleDefault alternatives @Platform {
      @ASIC -> @ModuleASIC
    } ()
  }
}
