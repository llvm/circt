// RUN: circt-opt %s | FileCheck %s

// CHECK: component @MyComponent (arg0: i64, arg1: i16) -> (arg2: i32) {
calyx.component @MyComponent (%arg0: i64, %arg1: i16) -> (%arg2: i32) {


// CHECK-NEXT: }
}
