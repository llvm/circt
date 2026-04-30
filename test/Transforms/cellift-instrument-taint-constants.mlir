// RUN: circt-opt --cellift-instrument="taint-constants=true" --split-input-file %s | FileCheck %s

// Test: Constants get all-ones taint when --taint-constants is set.
// CHECK-LABEL: hw.module @test_taint_constants
// CHECK-SAME: (out y : i8, out y_t : i8)
hw.module @test_taint_constants(out y : i8) {
  // CHECK: %c42_i8 = hw.constant 42 : i8
  // CHECK: %c-1_i8 = hw.constant -1 : i8
  // CHECK: hw.output %c42_i8, %c-1_i8 : i8, i8
  %c42 = hw.constant 42 : i8
  hw.output %c42 : i8
}
