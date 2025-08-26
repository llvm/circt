// RUN: circt-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: hw.module @plusargs_value
hw.module @plusargs_value() {
  // CHECK: sim.plusargs.test "foo"
  %0 = sim.plusargs.test "foo"
  // CHECK: sim.plusargs.value "bar" : i5
  %1, %2 = sim.plusargs.value "bar" : i5
}

// CHECK-LABEL: sim.func.dpi @dpi(out arg0 : i1, in %arg1 : i1, out arg2 : i1)
sim.func.dpi @dpi(out arg0: i1, in %arg1: i1, out arg2: i1)
func.func private @func(%arg1: i1) -> (i1, i1)

// CHECK-LABEL: hw.module @dpi_call
hw.module @dpi_call(in %clock : !seq.clock, in %enable : i1, in %in: i1) {
  // CHECK: sim.func.dpi.call @dpi(%in) clock %clock enable %enable : (i1) -> (i1, i1)
  %0, %1 = sim.func.dpi.call @dpi(%in) clock %clock enable %enable: (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @dpi(%in) clock %clock : (i1) -> (i1, i1)
  %2, %3 = sim.func.dpi.call @dpi(%in) clock %clock : (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @func(%in) enable %enable : (i1) -> (i1, i1)
  %4, %5 = sim.func.dpi.call @func(%in) enable %enable : (i1) -> (i1, i1)
  // CHECK: sim.func.dpi.call @func(%in) : (i1) -> (i1, i1)
  %6, %7 = sim.func.dpi.call @func(%in) : (i1) -> (i1, i1)
}

// CHECK-LABEL: hw.module @GraphSimulationControl
hw.module @GraphSimulationControl(in %clock: !seq.clock, in %en: i1) {
  // CHECK: sim.clocked_exit %clock, %en, success, verbose
  sim.clocked_exit %clock, %en, success, verbose
  // CHECK: sim.clocked_exit %clock, %en, success, quiet
  sim.clocked_exit %clock, %en, success, quiet
  // CHECK: sim.clocked_exit %clock, %en, failure, verbose
  sim.clocked_exit %clock, %en, failure, verbose
  // CHECK: sim.clocked_exit %clock, %en, failure, quiet
  sim.clocked_exit %clock, %en, failure, quiet

  // CHECK: sim.clocked_pause %clock, %en, verbose
  sim.clocked_pause %clock, %en, verbose
  // CHECK: sim.clocked_pause %clock, %en, quiet
  sim.clocked_pause %clock, %en, quiet
}

// CHECK-LABEL: func.func @SimulationControl
func.func @SimulationControl() {
  // CHECK: sim.exit success, verbose
  sim.exit success, verbose
  // CHECK: sim.exit success, quiet
  sim.exit success, quiet
  // CHECK: sim.exit failure, verbose
  sim.exit failure, verbose
  // CHECK: sim.exit failure, quiet
  sim.exit failure, quiet

  // CHECK: sim.pause verbose
  sim.pause verbose
  // CHECK: sim.pause quiet
  sim.pause quiet
  return
}
