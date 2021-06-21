// RUN: circt-opt %s -verify-diagnostics | FileCheck %s

// CHECK-LABEL: component ComponentWithInAndOutPorts (in1: 64, in2: 16) -> (out1: 32, out2: 8) {
calyx.component @ComponentWithInAndOutPorts (%in1: i64, %in2: i16) -> (%out1: i32, %out2: i8) {

// CHECK: }
}

// CHECK-LABEL: component ComponentWithInPort (x: 64) -> () {
calyx.component @ComponentWithInPort (%x: i64) -> () {}

// CHECK-LABEL: component ComponentWithOutPort () -> (y: 64) {
calyx.component @ComponentWithOutPort () -> (%y: i64) {}

// CHECK-LABEL: component ComponentWithNoPorts () -> () {
calyx.component @ComponentWithNoPorts () -> () {}
