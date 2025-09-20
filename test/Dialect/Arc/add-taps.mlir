// RUN: circt-opt %s --arc-add-taps | FileCheck %s

// CHECK-LABEL: hw.module @ObservePorts
hw.module @ObservePorts(in %x: i4, in %y: i4, out u: i4, out v: i4) {
  // CHECK-NEXT: arc.tap %x {names = ["x"]} : i4
  // CHECK-NEXT: arc.tap %y {names = ["y"]} : i4
  // CHECK-NEXT: %0 = comb.add
  // CHECK-NEXT: %1 = comb.sub
  %0 = comb.add %x, %y : i4
  %1 = comb.sub %x, %y : i4
  // CHECK-NEXT: arc.tap %0 {names = ["u"]} : i4
  // CHECK-NEXT: arc.tap %1 {names = ["v"]} : i4
  // CHECK-NEXT: hw.output
  hw.output %0, %1 : i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: hw.module @ObserveWires
hw.module @ObserveWires() {
  // CHECK-NEXT: arc.tap [[RD:%.+]] {names = ["x"]} : i4
  // CHECK-NEXT: %x = sv.wire
  // CHECK-NEXT: [[RD]] = sv.read_inout %x
  %x = sv.wire : !hw.inout<i4>
  %0 = sv.read_inout %x : !hw.inout<i4>

  // CHECK-NEXT: [[RD:%.+]] = sv.read_inout %y
  // CHECK-NEXT: arc.tap [[RD]] {names = ["y"]} : i4
  // CHECK-NEXT: %y = sv.wire
  %y = sv.wire : !hw.inout<i4>

  // CHECK-NEXT: hw.constant
  // CHECK-NEXT: arc.tap %c0_i4 {names = ["z"]} : i4
  // CHECK-NOT: hw.wire
  %c0_i4 = hw.constant 0 : i4
  %z = hw.wire %c0_i4 : i4

  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

// CHECK-LABEL: hw.module @Clocks
hw.module @Clocks(in %clk: !seq.clock) {
  // CHECK-NEXT: [[CAST:%.+]] = seq.from_clock %clk
  // CHECK-NEXT: arc.tap [[CAST]] {names = ["clk"]} : i1
}
