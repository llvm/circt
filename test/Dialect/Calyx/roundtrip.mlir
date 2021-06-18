// RUN: circt-opt %s | FileCheck %s

// CHECK: component @MyComponent(in1: 32, in2: 64) -> (out1: 42) {}
calyx.component @MyComponent({in1 = 32, in2 = 64}) -> ({out1 = 42}) {}
