// REQUIRES: verilator
// RUN: circt-translate %s -emit-verilog -verify-diagnostics > %t1.sv
// RUN: verilator --lint-only --top-module AB %t1.sv

module {
  rtl.module @B(%a: i1 { rtl.inout }) -> (i1 {rtl.name = "b"}, i1 {rtl.name = "c"}) {
    %0 = rtl.or %a, %a : i1
    %1 = rtl.and %a, %a : i1
    rtl.output %0, %1 : i1, i1
  }

  rtl.module @A(%d: i1, %e: i1) -> (i1 {rtl.name = "f"}) {
    %1 = rtl.mux %d, %d, %e : i1
    rtl.output %1 : i1
  }

  rtl.module @AAA(%d: i1, %e: i1) -> (i1 {rtl.name = "f"}) {
    %z = rtl.constant ( 0 : i1 ) : i1
    rtl.output %z : i1
  }

  rtl.module @AB(%w: i1, %x: i1) -> (i1 {rtl.name = "y"}, i1 {rtl.name = "z"}) {
    %w2 = rtl.instance "a1" @AAA(%w, %w1) : (i1, i1) -> (i1)
    %w1, %y = rtl.instance "b1" @B(%w2) : (i1) -> (i1, i1)
    rtl.output %y, %x : i1, i1
  }

  rtl.module @shl(%a: i1) -> (i1 {rtl.name = "b"}) {
    %0 = rtl.shl %a, %a : i1
    rtl.output %0 : i1
  }
}
