// RUN: circt-standalone-opt %s --circt-standalone-rename-wires | FileCheck %s

hw.module @foo(in %a: i32, out out: i32) {
  // CHECK: %foo_0 = hw.wire %a
  // CHECK: %foo_1 = hw.wire %foo_0
  %wire_a = hw.wire %a name "wire_a" : i32
  %wire_b = hw.wire %wire_a name "wire_b" : i32
  hw.output %wire_b : i32
}
