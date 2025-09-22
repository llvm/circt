// RUN: circt-opt %s --arc-merge-taps --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: hw.module @MergeTaps
hw.module @MergeTaps(in %in0: i4, in %in1: i4, in %in2: i4) {
  // CHECK:      arc.tap %in0 {names = ["in0Tap0", "in0Tap1", "in0Tap2"]} : i4
  // CHECK-NEXT: arc.tap %in1 {names = ["in1Tap0"]} : i4
  // CHECK-NEXT: arc.tap %in2 {names = ["in2Tap0", "in2Tap1"]} : i4
  // CHECK-NOT:  arc.tap
  arc.tap %in0 {names = ["in0Tap0"]} : i4
  arc.tap %in0 {names = ["in0Tap1"]} : i4
  arc.tap %in1 {names = ["in1Tap0"]} : i4
  arc.tap %in2 {names = ["in2Tap0"]} : i4
  arc.tap %in0 {names = ["in0Tap2"]} : i4
  arc.tap %in2 {names = ["in2Tap1"]} : i4
}

// CHECK-LABEL: hw.module @NoDuplicates
hw.module @NoDuplicates(in %in: i4) {
  // CHECK:     arc.tap %in {names = ["A", "B", "Tap0", "Tap1", "Tap2", "Tap3"]} : i4
  // CHECK-NOT: arc.tap
  arc.tap %in {names = ["Tap0", "A"]} : i4
  arc.tap %in {names = ["Tap1", "B"]} : i4
  arc.tap %in {names = ["Tap2", "A"]} : i4
  arc.tap %in {names = ["B"]} : i4
  arc.tap %in {names = ["Tap3"]} : i4
}

// CHECK-LABEL: hw.module @SameBlockOnly
hw.module @SameBlockOnly(in %in: i4) {
  // CHECK: arc.tap %in {names = ["Outer0", "Outer1"]} : i4
  // CHECK: arc.tap %in {names = ["Inner"]} : i4
  arc.tap %in {names = ["Outer0"]} : i4
  "foo.region"() ({
    ^bb0:
    arc.tap %in {names = ["Inner"]} : i4
  }) : () -> ()
  arc.tap %in {names = ["Outer1"]} : i4
}

// CHECK-LABEL: hw.module @CaseSensitive
hw.module @CaseSensitive(in %in: i4) {
  // CHECK: arc.tap %in {names = ["A", "a"]} : i4
  arc.tap %in {names = ["a"]} : i4
  arc.tap %in {names = ["A"]} : i4
}
