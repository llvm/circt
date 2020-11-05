// RUN: circt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

module {
  rtl.externmodule @E(%a: i1 {rtl.direction = "in"}, 
                %b: i1 {rtl.direction = "out"}, 
                %c: i1 {rtl.direction = "out"})

  // CHECK-LABEL: // external module E

  rtl.module @B(%a: i1 { rtl.inout }) -> (i1 {rtl.name = "b"}, i1 {rtl.name = "c"}) {
    %0 = rtl.or %a, %a : i1
    %1 = rtl.and %a, %a : i1
    rtl.output %0, %1 : i1, i1
  }

  // CHECK-LABEL: module B(
  // CHECK-NEXT:   inout  a,
  // CHECK-NEXT:   output b, c);
  // CHECK-EMPTY: 
  // CHECK-NEXT:   assign b = a | a;
  // CHECK-NEXT:   assign c = a & a;
  // CHECK-NEXT: endmodule

  rtl.module @A(%d: i1, %e: i1) -> (i1 {rtl.name = "f"}) {
    %1 = rtl.mux %d, %d, %e : i1
    rtl.output %1 : i1
  }

  // CHECK-LABEL: module A(
  // CHECK-NEXT:  input  d, e,
  // CHECK-NEXT:  output f);
  // CHECK-EMPTY:
  // CHECK-NEXT:  assign f = d ? d : e;
  // CHECK-NEXT: endmodule

  rtl.module @AAA(%d: i1, %e: i1) -> (i1 {rtl.name = "f"}) {
    %z = rtl.constant ( 0 : i1 ) : i1
    rtl.output %z : i1
  }

  // CHECK-LABEL: module AAA(
  // CHECK-NEXT:  input  d, e,
  // CHECK-NEXT:  output f);
  // CHECK-EMPTY:
  // CHECK-NEXT:  assign f = 1'h0;
  // CHECK-NEXT: endmodule

  rtl.module @AB(%w: i1, %x: i1) -> (i1 {rtl.name = "y"}, i1 {rtl.name = "z"}) {
    %w2 = rtl.instance "a1" @AAA(%w, %w1) : (i1, i1) -> (i1)
    %w1, %y = rtl.instance "b1" @B(%w2) : (i1) -> (i1, i1)
    rtl.output %y, %x : i1, i1
  }
  //CHECK-LABEL: module AB(
  //CHECK-NEXT:   input  w, x,
  //CHECK-NEXT:   output y, z);
  //CHECK-EMPTY: 
  //CHECK-NEXT:   wire w2;
  //CHECK-NEXT:   wire w1;
  //CHECK-NEXT:   wire y_0;
  //CHECK-EMPTY: 
  //CHECK-NEXT: A a1 (
  //CHECK-NEXT:     .d (w),
  //CHECK-NEXT:     .e (w1),
  //CHECK-NEXT:     .f (w2)
  //CHECK-NEXT:   )
  //CHECK-NEXT: B b1 (
  //CHECK-NEXT:     .a (w2),
  //CHECK-NEXT:     .b (w1),
  //CHECK-NEXT:     .c (y_0)
  //CHECK-NEXT:   )
  //CHECK-NEXT:   assign y = y_0;
  //CHECK-NEXT:   assign z = x;
  //CHECK-NEXT: endmodule

  rtl.module @shl(%a: i1) -> (i1 {rtl.name = "b"}) {
    %0 = rtl.shl %a, %a : i1
    rtl.output %0 : i1
  }

  // CHECK-LABEL:  module shl(
  // CHECK-NEXT:   input  a,
  // CHECK-NEXT:   output b);
  // CHECK-EMPTY:
  // CHECK-NEXT:   assign b = a << a;
  // CHECK-NEXT: endmodule
}
