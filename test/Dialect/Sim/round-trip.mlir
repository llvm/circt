// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s


// CHECK-LABEL: hw.module @plusargs_value
hw.module @plusargs_value() {
  // CHECK: sim.plusargs.test "foo"
  %0 = sim.plusargs.test "foo"
  // CHECK: sim.plusargs.value "bar" : i5
  %1, %2 = sim.plusargs.value "bar" : i5
}

