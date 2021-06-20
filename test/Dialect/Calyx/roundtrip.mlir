// RUN: circt-opt %s | FileCheck %s

// CHECK: component @MyComponent (in1: 64, in2: 16) -> (out1: 32, out2: 8) {
calyx.component @MyComponent (%in1: i64, %in2: i16) -> (%out1: i32, %out2: i8) {

// CHECK: }
}
