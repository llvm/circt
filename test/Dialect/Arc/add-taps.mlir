// RUN: circt-opt %s --arc-add-taps | FileCheck %s

// CHECK-LABEL: hw.module @ObservePorts
hw.module @ObservePorts(%x: i4, %y: i4) -> (u: i4, v: i4) {
  // CHECK-NEXT: arc.tap %x : i4 input r "x" : i4
  // CHECK-NEXT: arc.tap %y : i4 input r "y" : i4
  // CHECK-NEXT: [[V0:%.+]] = comb.add
  // CHECK-NEXT: [[V1:%.+]] = comb.sub
  %0 = comb.add %x, %y : i4
  %1 = comb.sub %x, %y : i4
  // CHECK-NEXT: arc.tap [[V0]] : i4 output r "u" : i4
  // CHECK-NEXT: arc.tap [[V1]] : i4 output r "v" : i4
  // CHECK-NEXT: hw.output
  hw.output %0, %1 : i4, i4
}
// CHECK-NEXT: }


// CHECK-LABEL: hw.module @ObserveWires
hw.module @ObserveWires() {
  // CHECK-NEXT: %x = sv.wire
  // CHECK-NEXT: [[RD:%.+]] = sv.read_inout %x
  // CHECK-NEXT: arc.tap [[RD]] : i4 wire r "x" : i4
  %x = sv.wire : !hw.inout<i4>
  %0 = sv.read_inout %x : !hw.inout<i4>

  // CHECK-NEXT: %y = sv.wire
  // CHECK-NEXT: [[RD:%.+]] = sv.read_inout %y
  // CHECK-NEXT: arc.tap [[RD]] : i4 wire r "y" : i4
  %y = sv.wire : !hw.inout<i4>

  // CHECK-NEXT: hw.constant
  // CHECK-NEXT: %z = hw.wire
  // CHECK-NEXT: arc.tap %z : i4 wire r "z" : i4
  %c0_i4 = hw.constant 0 : i4
  %z = hw.wire %c0_i4 : i4

  // CHECK-NEXT: hw.output
}
// CHECK-NEXT: }

