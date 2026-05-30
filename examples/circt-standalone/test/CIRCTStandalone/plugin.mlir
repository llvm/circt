// UNSUPPORTED: system-windows
// UNSUPPORTED: no-circt-standalone-plugin
// RUN: circt-opt %s --load-dialect-plugin=%circt_standalone_libs/CIRCTStandalonePlugin%shlibext --pass-pipeline="builtin.module(hw.module(circt-standalone-rename-wires))" | FileCheck %s

hw.module @foo(in %a: i32, out out: i32) {
  // CHECK: %foo_0 = hw.wire %a
  %wire_a = hw.wire %a name "wire_a" : i32
  hw.output %wire_a : i32
}
