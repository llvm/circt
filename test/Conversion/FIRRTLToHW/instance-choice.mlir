// RUN: circt-opt --lower-firrtl-to-hw -split-input-file %s | FileCheck %s
// RUN: circt-opt --lower-firrtl-to-hw="disallow-instance-choice-default" -split-input-file %s | FileCheck %s --check-prefix=NODEFAULT

// Test basic instance choice lowering with single option
// CHECK-LABEL: hw.module @SingleOption
firrtl.circuit "SingleOption" {
  sv.macro.decl @targets$Platform$FPGA
  firrtl.option @Platform {
    firrtl.option_case @FPGA { case_macro = @targets$Platform$FPGA }
  }

  sv.macro.decl @targets$Platform$SingleOption$inst

  firrtl.module private @DefaultMod() {}
  firrtl.module private @FPGAMod() {}

  firrtl.module @SingleOption() {
    // CHECK: hw.instance "inst_default" sym @{{.+}} @DefaultMod
    // NODEFAULT: sv.error "No valid instance choice case for option Platform, set a macro one of [targets$Platform$FPGA]"
    firrtl.instance_choice inst {instance_macro = @targets$Platform$SingleOption$inst} @DefaultMod alternatives @Platform { @FPGA -> @FPGAMod } ()
  }
}
