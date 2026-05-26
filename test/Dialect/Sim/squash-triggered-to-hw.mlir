// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(sim-squash-triggered{convert-to-hw}))' --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: hw.module @same_condition
// CHECK: %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT: hw.triggered posedge %[[TRG]](%en) : i1 {
// CHECK-NEXT: ^bb0(%[[EN:.*]]: i1):
// CHECK-NEXT:   scf.if %[[EN]] {
// CHECK-NEXT:     "test.a"() : () -> ()
// CHECK-NEXT:     "test.b"() : () -> ()
// CHECK-NEXT:   }
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
// CHECK: %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT: hw.triggered posedge %[[TRG]](%ena, %enb) : i1, i1 {
// CHECK-NEXT: ^bb0(%[[ENA:.*]]: i1, %[[ENB:.*]]: i1):
// CHECK-NEXT:   scf.if %[[ENA]] {
// CHECK-NEXT:     "test.a"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.if %[[ENB]] {
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

// CHECK-LABEL: hw.module @fstring_capture
// CHECK: %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT: hw.triggered posedge %[[TRG]](%{{.*}}) : !sim.fstring {
// CHECK-NEXT: ^bb0(%[[MSG:.*]]: !sim.fstring):
// CHECK-NEXT:   sim.proc.print %[[MSG]]
// CHECK-NEXT: }
hw.module @fstring_capture(in %clk : !seq.clock) {
  %prefix = sim.fmt.literal "hello"
  %msg = sim.fmt.concat (%prefix)
  sim.triggered %clk {
    sim.proc.print %msg
  }
}
