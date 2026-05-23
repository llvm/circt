// RUN: circt-opt --sim-convert-triggered-to-hw --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: hw.module @simple
// CHECK: %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT: hw.triggered posedge %[[TRG]](%value, %en) : i8, i1 {
// CHECK-NEXT: ^bb0(%[[VALUE:.*]]: i8, %[[EN:.*]]: i1):
// CHECK-NEXT:   scf.if %[[EN]] {
// CHECK-NEXT:     "test.use"(%[[VALUE]]) : (i8) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
hw.module @simple(in %clk : !seq.clock, in %en : i1, in %value : i8) {
  sim.triggered %clk if %en {
    "test.use"(%value) : (i8) -> ()
  }
}

// -----

// CHECK-LABEL: hw.module @nested_if
// CHECK: %[[TRG:.*]] = seq.from_clock %clk
// CHECK-NEXT: hw.triggered posedge %[[TRG]](%value, %cond) : i8, i1 {
// CHECK-NEXT: ^bb0(%[[VALUE:.*]]: i8, %[[COND:.*]]: i1):
// CHECK-NEXT:   scf.if %[[COND]] {
// CHECK-NEXT:     "test.use"(%[[VALUE]]) : (i8) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
hw.module @nested_if(in %clk : !seq.clock, in %cond : i1, in %value : i8) {
  sim.triggered %clk {
    scf.if %cond {
      "test.use"(%value) : (i8) -> ()
    }
  }
}
