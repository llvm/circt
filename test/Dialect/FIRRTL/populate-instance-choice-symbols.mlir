// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-populate-instance-choice-symbols))' %s | FileCheck %s

firrtl.circuit "Top" {
  // CHECK: sv.macro.decl @targets$Platform$FPGA
  // CHECK: sv.macro.decl @targets$Platform$Top$inst
  // CHECK: sv.macro.decl @targets$Platform$AnotherTop$inst

  // CHECK: firrtl.option @Platform {
  // CHECK-NEXT: firrtl.option_case @FPGA {case_macro = @targets$Platform$FPGA}
  // CHECK-NEXT: firrtl.option_case @ASIC {case_macro = @targets$Platform$ASIC}
  firrtl.option @Platform {
    firrtl.option_case @FPGA
    firrtl.option_case @ASIC
  }

  sv.macro.decl @targets$Platform$ASIC

  firrtl.module private @ModuleDefault() { }
  firrtl.module private @ModuleFPGA() { }

  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock) {
    // CHECK: firrtl.instance_choice inst {instance_macro = @targets$Platform$Top$inst} @ModuleDefault alternatives @Platform
    firrtl.instance_choice inst @ModuleDefault alternatives @Platform {
      @FPGA -> @ModuleFPGA,
      @ASIC -> @ModuleDefault
    } ()
  }
  // CHECK-LABEL: firrtl.module @AnotherTop
  firrtl.module @AnotherTop(in %clock: !firrtl.clock) {
    // CHECK: firrtl.instance_choice inst {instance_macro = @targets$Platform$AnotherTop$inst} @ModuleDefault alternatives @Platform
    firrtl.instance_choice inst @ModuleDefault alternatives @Platform {
      @FPGA -> @ModuleFPGA,
      @ASIC -> @ModuleDefault
    } ()
  }
}
