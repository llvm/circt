// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s


// CHECK-LABEL: hw.module @plusargs_value
hw.module @plusargs_value() {
  // CHECK: sim.plusargs.test "foo"
  %0 = sim.plusargs.test "foo"
  // CHECK: sim.plusargs.value "bar" : i5
  %1, %2 = sim.plusargs.value "bar" : i5
}

// CHECK-LABEL: hw.module @stop_finish
hw.module @stop_finish(in %clock : !seq.clock, in %cond : i1) {
  // CHECK: sim.finish %clock, %cond
  sim.finish %clock, %cond
  // CHECK: sim.fatal %clock, %cond
  sim.fatal %clock, %cond
}

// CHECK-LABEL: sim.func.dpi @dpi(out arg0 : i1, in %arg1 : i1, out arg2 : i1)
sim.func.dpi @dpi(out arg0: i1, in %arg1: i1, out arg2: i1)

hw.module @dpi_call(in %clock : !seq.clock, in %enable : i1, in %in: i1) {
  // CHECK: sim.func.dpi.call @dpi(%in) clock %clock enable %enable : (i1) -> (i1, i1)
  %0, %1 = sim.func.dpi.call @dpi(%in) clock %clock enable %enable: (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @dpi(%in) clock %clock : (i1) -> (i1, i1)
  %2, %3 = sim.func.dpi.call @dpi(%in) clock %clock : (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @dpi(%in) enable %enable : (i1) -> (i1, i1)
  %4, %5 = sim.func.dpi.call @dpi(%in) enable %enable : (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @dpi(%in) : (i1) -> (i1, i1)
  %6, %7 = sim.func.dpi.call @dpi(%in) : (i1) -> (i1, i1)
}
