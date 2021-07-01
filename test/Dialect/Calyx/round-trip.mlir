// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK: calyx.program {
calyx.program {

  // CHECK-LABEL:  calyx.component @ComponentWithInAndOutPorts(%in1: i32, %in2: i16) -> (%out1: i32, %out2: i8) {
  calyx.component @ComponentWithInAndOutPorts(%in1: i32, %in2: i16) -> (%out1: i32, %out2: i8) {
    // CHECK:        calyx.wires {
    // CHECK:        calyx.control {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @ComponentWithInPort(%x: i64) -> () {
  calyx.component @ComponentWithInPort(%x: i64) -> () {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @ComponentWithOutPort() -> (%y: i64) {
  calyx.component @ComponentWithOutPort() -> (%y: i64) {
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @ComponentWithNoPorts() -> () {
  calyx.component @ComponentWithNoPorts() -> () {
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @A(%in: i8) -> (%out: i8) {
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @main() -> () {
    %in1, %out1 = calyx.cell "c0" @A : i8, i8
    %in2, %out2 = calyx.cell "c1" @A : i8, i8
    %c1_i1 = constant 1 : i1

    calyx.wires {
      // CHECK: calyx.group @SomeGroup {
      calyx.group @SomeGroup {
        // CHECK: calyx.assign %1#0 = %0#1 : i8
        // CHECK-NEXT: calyx.done %true : i1
        calyx.assign %in2 = %out1 : i8
        calyx.done %c1_i1 : i1
      }
    }
    calyx.control {}
  }
}
