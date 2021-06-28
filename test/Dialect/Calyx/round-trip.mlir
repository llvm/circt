// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK: calyx.program {
calyx.program {

  // CHECK-LABEL:  calyx.component @ComponentWithInAndOutPorts(%in1: i64, %in2: i16) -> (%out1: i32, %out2: i8) {
  calyx.component @ComponentWithInAndOutPorts(%in1: i64, %in2: i16) -> (%out1: i32, %out2: i8) {
    // CHECK:        calyx.cells {
    // CHECK:        calyx.wires {
    // CHECK:        calyx.control {
    calyx.cells {}
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @ComponentWithInPort(%x: i64) -> () {
  calyx.component @ComponentWithInPort(%x: i64) -> () {
    calyx.cells {}
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @ComponentWithOutPort() -> (%y: i64) {
  calyx.component @ComponentWithOutPort() -> (%y: i64) {
    calyx.cells {}
    calyx.wires {}
    calyx.control {}
  }

  // CHECK-LABEL: calyx.component @ComponentWithNoPorts() -> () {
  calyx.component @ComponentWithNoPorts() -> () {
    calyx.cells {}
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @main() -> () {
    calyx.cells {
      // CHECK: %0:4 = calyx.cell "c0" @ComponentWithInAndOutPorts : i64, i16, i32, i8
      // CHECK-NEXT: calyx.cell "c1" @ComponentWithNoPorts
      %in1, %in2, %out1, %out2 = calyx.cell "c0" @ComponentWithInAndOutPorts : i64, i16, i32, i8
      calyx.cell "c1" @ComponentWithNoPorts
    }
    calyx.wires {
      // CHECK: calyx.group @SomeGroup {
      calyx.group @SomeGroup {}
    }
    calyx.control {}
  }

}
