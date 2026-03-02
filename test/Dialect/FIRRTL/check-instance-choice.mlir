// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-check-instance-choice{dump-info=true}))'  --verify-diagnostics %s 2>&1 | FileCheck %s

// Test case demonstrating instance choice scenarios with multiple public modules
firrtl.circuit "Top" {
  firrtl.option @Platform {
    firrtl.option_case @FPGA
    firrtl.option_case @ASIC
  }
  firrtl.option @Optimization {
    firrtl.option_case @Speed
  }

  // Leaf modules
  firrtl.module private @Shared() { }
  firrtl.module private @LeafA() { }
  firrtl.module private @TargetX() { }
  firrtl.module private @TargetY() { }
  firrtl.module private @TargetZ() { }

  // Same module with different options
  firrtl.module private @Middle() {
    firrtl.instance_choice leaf @LeafA alternatives @Optimization { @Speed -> @LeafA } ()
  }

  // Different target modules for each case
  firrtl.module private @MultiTarget() {
    firrtl.instance_choice target @TargetX alternatives @Platform { @FPGA -> @TargetY, @ASIC -> @TargetZ } ()
  }

  // CHECK-LABEL: Public module: Top
  // CHECK-NEXT:   -> Top: <always>
  // CHECK-NEXT:   -> Shared: <always>
  // CHECK-NEXT:   -> Middle: <always>
  // CHECK-NEXT:   -> LeafA: Optimization=<default>, Optimization=Speed
  firrtl.module public @Top() {
    firrtl.instance m @Middle()
    firrtl.instance s @Shared()
  }

  // CHECK-LABEL: Public module: AltTop
  // CHECK-NEXT:   -> AltTop: <always>
  // CHECK-NEXT:   -> MultiTarget: <always>
  // CHECK-NEXT:   -> TargetZ: <always>
  // CHECK-NEXT:   -> TargetY: Platform=FPGA
  // CHECK-NEXT:   -> TargetX: Platform=<default>
  firrtl.module public @AltTop() {
    firrtl.instance multi @MultiTarget()
    firrtl.instance z @TargetZ()
  }
}
