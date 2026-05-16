// RUN: circt-opt --lower-sim-to-sv --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: hw.module @simple_triggered
hw.module @simple_triggered(in %clock : !seq.clock) {
  sim.triggered %clock {
    "some.user"() : () -> ()
  }

  // CHECK: %[[CLOCK:.*]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLOCK]] {
  // CHECK-NEXT:   "some.user"() : () -> ()
  // CHECK-NEXT: }
}

// CHECK-LABEL: hw.module @conditional_triggered
hw.module @conditional_triggered(
    in %clock : !seq.clock, in %en : i1) {
  sim.triggered %clock if %en {
    "some.user"() : () -> ()
  }

  // CHECK: %[[CLOCK:.*]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLOCK]] {
  // CHECK-NEXT:   sv.if %en {
  // CHECK-NEXT:     "some.user"() : () -> ()
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
}

// CHECK-LABEL: hw.module @multiple_triggered
hw.module @multiple_triggered(
    in %clock : !seq.clock, in %en : i1) {

  sim.triggered %clock {
    "some.user"() : () -> ()
  }

  sim.triggered %clock if %en {
    "some.user"() : () -> ()
  }

  // CHECK: %[[CLOCK0:.*]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLOCK0]] {
  // CHECK-NEXT:   "some.user"() : () -> ()
  // CHECK-NEXT: }
  // CHECK: %[[CLOCK1:.*]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLOCK1]] {
  // CHECK-NEXT:   sv.if %en {
  // CHECK-NEXT:     "some.user"() : () -> ()
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
}
