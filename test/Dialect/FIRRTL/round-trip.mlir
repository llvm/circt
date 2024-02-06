// RUN: circt-opt %s | circt-opt | FileCheck %s
// Basic MLIR operation parser round-tripping

firrtl.circuit "Basic" {
firrtl.extmodule @Basic()

// CHECK-LABEL: firrtl.module @Intrinsics
firrtl.module @Intrinsics(in %ui : !firrtl.uint, in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
  // CHECK-NEXT: firrtl.int.sizeof %ui : (!firrtl.uint) -> !firrtl.uint<32>
  %size = firrtl.int.sizeof %ui : (!firrtl.uint) -> !firrtl.uint<32>

  // CHECK-NEXT: firrtl.int.isX %ui : !firrtl.uint
  %isx = firrtl.int.isX %ui : !firrtl.uint

  // CHECK-NEXT: firrtl.int.plusargs.test "foo"
  // CHECK-NEXT: firrtl.int.plusargs.value "bar" : !firrtl.uint<5>
  %foo_found = firrtl.int.plusargs.test "foo"
  %bar_found, %bar_value = firrtl.int.plusargs.value "bar" : !firrtl.uint<5>

  // CHECK-NEXT: firrtl.int.clock_gate %clock, %ui1
  // CHECK-NEXT: firrtl.int.clock_gate %clock, %ui1, %ui1
  %cg0 = firrtl.int.clock_gate %clock, %ui1
  %cg1 = firrtl.int.clock_gate %clock, %ui1, %ui1
}

// CHECK-LABEL: firrtl.module @FPGAProbe
firrtl.module @FPGAProbe(
  in %clock: !firrtl.clock,
  in %reset: !firrtl.uint<1>,
  in %in: !firrtl.uint<8>
) {
  // CHECK: firrtl.int.fpga_probe %clock, %in : !firrtl.uint<8>
  firrtl.int.fpga_probe %clock, %in : !firrtl.uint<8>
}

// CHECK-LABEL: firrtl.option @Platform
firrtl.option @Platform {
  // CHECK:firrtl.option_case @FPGA
  firrtl.option_case @FPGA
  // CHECK:firrtl.option_case @ASIC
  firrtl.option_case @ASIC
}

firrtl.module private @DefaultTarget(in %clock: !firrtl.clock) {}
firrtl.module private @FPGATarget(in %clock: !firrtl.clock) {}
firrtl.module private @ASICTarget(in %clock: !firrtl.clock) {}

// CHECK-LABEL: firrtl.module @Foo
firrtl.module @Foo(in %clock: !firrtl.clock) {
  // CHECK: %inst_clock = firrtl.instance_choice inst interesting_name @DefaultTarget alternatives @Platform
  // CHECK-SAME: { @FPGA -> @FPGATarget, @ASIC -> @ASICTarget } (in clock: !firrtl.clock)
  %inst_clock = firrtl.instance_choice inst interesting_name @DefaultTarget alternatives @Platform
    { @FPGA -> @FPGATarget, @ASIC -> @ASICTarget } (in clock: !firrtl.clock)
  firrtl.strictconnect %inst_clock, %clock : !firrtl.clock
}

firrtl.layer @LayerA bind {
  firrtl.layer @LayerB bind {}
}

// CHECK-LABEL: firrtl.module @Layers
// CHECK-SAME:    out %a: !firrtl.probe<uint<1>, @LayerA>
// CHECK-SAME:    out %b: !firrtl.rwprobe<uint<1>, @LayerA::@LayerB>
firrtl.module @Layers(
  out %a: !firrtl.probe<uint<1>, @LayerA>,
  out %b: !firrtl.rwprobe<uint<1>, @LayerA::@LayerB>
) {}

// CHECK-LABEL: firrtl.module @LayersEnabled
// CHECK-SAME:    layers = [@LayerA]
firrtl.module @LayersEnabled() attributes {layers = [@LayerA]} {
}

// CHECK-LABEL: firrtl.module @PropertyArithmetic
firrtl.module @PropertyArithmetic() {
  %0 = firrtl.integer 1
  %1 = firrtl.integer 2

  // CHECK: firrtl.integer.add %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  %2 = firrtl.integer.add %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

  // CHECK: firrtl.integer.mul %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  %3 = firrtl.integer.mul %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer

  // CHECK: firrtl.integer.shr %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
  %4 = firrtl.integer.shr %0, %1 : (!firrtl.integer, !firrtl.integer) -> !firrtl.integer
}

}
