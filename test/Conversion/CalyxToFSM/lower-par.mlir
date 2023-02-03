// RUN: circt-opt --split-input-file -pass-pipeline='builtin.module(calyx.component(lower-calyx-to-fsm))' %s | FileCheck %s

calyx.component @main(%go: i1 {go}, %reset: i1 {reset}, %clk: i1 {clk}) -> (%done: i1 {done}) {
  %c2_i8 = hw.constant 2 : i8
  %c1_i8 = hw.constant 1 : i8
  %c0_i8 = hw.constant 0 : i8
  %true = hw.constant true
  %a.in, %a.write_en, %a.clk, %a.reset, %a.out, %a.done = calyx.register @a : i8, i1, i1, i1, i8, i1
  %b.in, %b.write_en, %b.clk, %b.reset, %b.out, %b.done = calyx.register @b : i8, i1, i1, i1, i8, i1
  %c.in, %c.write_en, %c.clk, %c.reset, %c.out, %c.done = calyx.register @c : i8, i1, i1, i1, i8, i1
  calyx.wires {
    %0 = calyx.undef : i1
    calyx.group @A {
      %A.go = calyx.group_go %0 : i1
      calyx.assign %a.in = %A.go ? %c0_i8 : i8
      calyx.assign %a.write_en = %A.go ? %true : i1
      calyx.group_done %a.done : i1
    }
    calyx.group @B {
      %B.go = calyx.group_go %0 : i1
      calyx.assign %b.in = %B.go ? %c1_i8 : i8
      calyx.assign %b.write_en = %B.go ? %true : i1
      calyx.group_done %b.done : i1
    }
    calyx.group @C {
      %C.go = calyx.group_go %0 : i1
      calyx.assign %c.in = %C.go ? %c2_i8 : i8
      calyx.assign %c.write_en = %C.go ? %true : i1
      calyx.group_done %c.done : i1
    }
  }
  calyx.control {
    calyx.par {
      calyx.enable @A
      calyx.enable @B
      calyx.enable @C
    }
  }
}
