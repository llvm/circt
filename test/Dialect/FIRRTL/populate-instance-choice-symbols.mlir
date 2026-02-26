// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-populate-instance-choice-symbols))' %s | FileCheck %s

firrtl.circuit "Top" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }

  firrtl.module private @ModuleDefault(in %clock: !firrtl.clock) { }
  firrtl.module private @ModuleFPGA(in %clock: !firrtl.clock) { }

  // CHECK: sv.macro.decl @__target_Platform_Top_inst
  // CHECK: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock) {
    // CHECK: firrtl.instance_choice inst {instance_macro = @__target_Platform_Top_inst} @ModuleDefault alternatives @Platform
    %inst_clock = firrtl.instance_choice inst @ModuleDefault alternatives @Platform {
      @FPGA -> @ModuleFPGA
    } (in clock: !firrtl.clock)
    firrtl.matchingconnect %inst_clock, %clock : !firrtl.clock
  }
}

