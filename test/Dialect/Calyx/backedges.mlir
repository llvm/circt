// RUN: circt-opt --split-input-file %s | FileCheck %s

calyx.program "main" {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %r2.in, %r2.write_en, %r2.clk, %r2.reset, %r2.out, %r2.done = calyx.register @r2 : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // CHECK: calyx.assign %r.in = %0 ? %true : i1
      // CHECK: %0 = comb.and %r.out, %r2.out : i1
      calyx.assign %r.in = %0 ? %c1_1 : i1
      %0 = comb.and %r.out, %r2.out : i1
    }
    calyx.control { }
  }
}
