// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-populate-instance-choice-symbols))' %s | FileCheck %s

firrtl.circuit "Top" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
  }

  firrtl.module private @ModuleDefault() { }
  firrtl.module private @ModuleFPGA() { }

  // CHECK: sv.macro.decl @__target_Platform_Top_inst
  // CHECK: sv.macro.decl @__target_Platform_AnotherTop_inst

  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock) {
    // CHECK: firrtl.instance_choice inst {instance_macro = @__target_Platform_Top_inst} @ModuleDefault alternatives @Platform
    firrtl.instance_choice inst @ModuleDefault alternatives @Platform {
      @FPGA -> @ModuleFPGA
    } ()
  }
  // CHECK-LABEL: firrtl.module @AnotherTop
  firrtl.module @AnotherTop(in %clock: !firrtl.clock) {
    // CHECK: firrtl.instance_choice inst {instance_macro = @__target_Platform_AnotherTop_inst} @ModuleDefault alternatives @Platform
    firrtl.instance_choice inst @ModuleDefault alternatives @Platform {
      @FPGA -> @ModuleFPGA
    } ()
  }
}

