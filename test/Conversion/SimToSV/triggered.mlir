// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

// CHECK-LABEL: hw.module @simple_triggered
hw.module @simple_triggered(in %clock : !seq.clock) {
  %msg = sim.fmt.literal "tick"
  sim.triggered %clock {
    sim.proc.print %msg
  }

  // CHECK: %[[CLOCK:.*]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLOCK]] {
  // CHECK-NEXT:   sv.write "tick"
  // CHECK-NEXT: }
}

// CHECK-LABEL: hw.module @conditional_triggered
hw.module @conditional_triggered(
    in %clock : !seq.clock, in %en : i1, in %val : i8) {
  %prefix = sim.fmt.literal "value="
  %value = sim.fmt.hex %val, isUpper false specifierWidth 2 : i8
  %msg = sim.fmt.concat (%prefix, %value)
  sim.triggered %clock if %en {
    sim.proc.print %msg
  }

  // CHECK: %[[CLOCK:.*]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLOCK]] {
  // CHECK-NEXT:   sv.if %en {
  // CHECK-NEXT:     sv.write "value=%02x"(%val) : i8
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
}

// CHECK-LABEL: hw.module @multiple_triggered
hw.module @multiple_triggered(
    in %clock : !seq.clock, in %en : i1, in %lhs : i8, in %rhs : i8) {
  %lhsPrefix = sim.fmt.literal "lhs="
  %lhsValue = sim.fmt.hex %lhs, isUpper false specifierWidth 2 : i8
  %lhsMsg = sim.fmt.concat (%lhsPrefix, %lhsValue)

  %rhsPrefix = sim.fmt.literal "rhs="
  %rhsValue = sim.fmt.hex %rhs, isUpper false specifierWidth 2 : i8
  %rhsMsg = sim.fmt.concat (%rhsPrefix, %rhsValue)

  sim.triggered %clock {
    sim.proc.print %lhsMsg
  }

  sim.triggered %clock if %en {
    sim.proc.print %rhsMsg
  }

  // CHECK: %[[CLOCK0:.*]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLOCK0]] {
  // CHECK-NEXT:   sv.write "lhs=%02x"(%lhs) : i8
  // CHECK-NEXT: }
  // CHECK: %[[CLOCK1:.*]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLOCK1]] {
  // CHECK-NEXT:   sv.if %en {
  // CHECK-NEXT:     sv.write "rhs=%02x"(%rhs) : i8
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
}
