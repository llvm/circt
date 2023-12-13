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

}
