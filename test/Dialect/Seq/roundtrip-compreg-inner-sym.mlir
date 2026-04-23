// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

hw.module @Foo(in %a: i42, in %clk: !seq.clock) {
  // CHECK: seq.compreg sym @"foo/bar" %a, %clk : i42
  %0 = seq.compreg sym @"foo/bar" %a, %clk : i42
  hw.output
}

hw.module @Bar(in %d: i32, in %clk: !seq.clock, in %en: i1) {
  // CHECK: seq.compreg.ce sym @"weird name" %d, %clk, %en : i32
  %1 = seq.compreg.ce sym @"weird name" %d, %clk, %en : i32
  hw.output
}

