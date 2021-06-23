// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL:  calyx.component @ComponentWithInAndOutPorts(%in1: i64, %in2: i16) -> (%out1: i32, %out2: i8) {
calyx.component @ComponentWithInAndOutPorts(%in1: i64, %in2: i16) -> (%out1: i32, %out2: i8) {

  // CHECK:        calyx.cells {
  calyx.cells {}
  // CHECK:        calyx.wires {
  calyx.wires {}
  // CHECK:        calyx.control {
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
