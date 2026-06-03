// RUN: circt-opt --sim-squash-triggered --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: hw.module @same_condition
// CHECK: sim.triggered %clk if %en {
// CHECK-NEXT:   "test.a"() : () -> ()
// CHECK-NEXT:   "test.b"() : () -> ()
// CHECK-NEXT: }
hw.module @same_condition(in %clk : !seq.clock, in %en : i1) {
  sim.triggered %clk if %en {
    "test.a"() : () -> ()
  }
  sim.triggered %clk if %en {
    "test.b"() : () -> ()
  }
}

// -----

// CHECK-LABEL: hw.module @different_conditions
// CHECK: sim.triggered %clk {
// CHECK-NEXT:   scf.if %ena {
// CHECK-NEXT:     "test.a"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.if %enb {
// CHECK-NEXT:     "test.b"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
hw.module @different_conditions(
    in %clk : !seq.clock, in %ena : i1, in %enb : i1) {
  sim.triggered %clk if %ena {
    "test.a"() : () -> ()
  }
  sim.triggered %clk if %enb {
    "test.b"() : () -> ()
  }
}

// -----

// CHECK-LABEL: hw.module @mixed_conditions
// CHECK: sim.triggered %clk {
// CHECK-NEXT:   "test.u0"() : () -> ()
// CHECK-NEXT:   scf.if %en {
// CHECK-NEXT:     "test.c1"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.u2"() : () -> ()
// CHECK-NEXT: }
hw.module @mixed_conditions(in %clk : !seq.clock, in %en : i1) {
  sim.triggered %clk {
    "test.u0"() : () -> ()
  }
  sim.triggered %clk if %en {
    "test.c1"() : () -> ()
  }
  sim.triggered %clk {
    "test.u2"() : () -> ()
  }
}

// -----

// CHECK-LABEL: hw.module @different_clocks
// CHECK: sim.triggered %clka {
// CHECK-NEXT:   "test.a"() : () -> ()
// CHECK-NEXT: }
// CHECK: sim.triggered %clkb if %en {
// CHECK-NEXT:   "test.b"() : () -> ()
// CHECK-NEXT: }
hw.module @different_clocks(
    in %clka : !seq.clock, in %clkb : !seq.clock, in %en : i1) {
  sim.triggered %clka {
    "test.a"() : () -> ()
  }
  sim.triggered %clkb if %en {
    "test.b"() : () -> ()
  }
}

// -----

// CHECK-LABEL: hw.module @interleaved_same_clock
// CHECK: sim.triggered %clkb if %en {
// CHECK-NEXT:   "test.b0"() : () -> ()
// CHECK-NEXT: }
// CHECK: sim.triggered %clka {
// CHECK-NEXT:   "test.a0"() : () -> ()
// CHECK-NEXT:   "test.a1"() : () -> ()
// CHECK-NEXT: }
hw.module @interleaved_same_clock(
    in %clka : !seq.clock, in %clkb : !seq.clock, in %en : i1) {
  sim.triggered %clka {
    "test.a0"() : () -> ()
  }
  sim.triggered %clkb if %en {
    "test.b0"() : () -> ()
  }
  sim.triggered %clka {
    "test.a1"() : () -> ()
  }
}

// -----

// CHECK-LABEL: hw.module @repeated_condition_segments
// CHECK: sim.triggered %clk {
// CHECK-NEXT:   scf.if %en {
// CHECK-NEXT:     "test.c0"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   "test.u0"() : () -> ()
// CHECK-NEXT:   scf.if %en {
// CHECK-NEXT:     "test.c1"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
hw.module @repeated_condition_segments(in %clk : !seq.clock, in %en : i1) {
  sim.triggered %clk if %en {
    "test.c0"() : () -> ()
  }
  sim.triggered %clk {
    "test.u0"() : () -> ()
  }
  sim.triggered %clk if %en {
    "test.c1"() : () -> ()
  }
}
