// RUN: circt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

module {
  rtl.externmodule @E(%a: i1 {rtl.direction = "in"}, 
                %b: i1 {rtl.direction = "out"}, 
                %c: i1 {rtl.direction = "out"})

  // CHECK-LABEL: // external module E

  rtl.module @B(%a: i1 {rtl.direction = "in"}, 
                %b: i1 {rtl.direction = "out"}, 
                %c: i1 {rtl.direction = "out"}) {
    %0 = rtl.or %a, %a : i1
    %1 = rtl.and %a, %a : i1
    rtl.connect %b, %0 : i1
    rtl.connect %c, %1 : i1
  }

  // CHECK-LABEL: module B(
  // CHECK-NEXT:   input  a,
  // CHECK-NEXT:   output b, c);
  // CHECK-EMPTY: 
  // CHECK-NEXT:   assign b = a | a;
  // CHECK-NEXT:   assign c = a & a;
  // CHECK-NEXT: endmodule

  rtl.module @A(%d: i1 {rtl.direction = "in"}, 
                %e: i1 {rtl.direction = "in"}, 
                %f: i1 {rtl.direction = "out"}) {
    %0 = rtl.and %d, %e : i1
    rtl.connect %f, %0 : i1

    %1 = rtl.mux %d, %d, %e : i1
    rtl.connect %f, %1 : i1
  }
  // CHECK-LABEL: module A(
  // CHECK-NEXT:  input  d, e,
  // CHECK-NEXT:  output f);
  // CHECK-EMPTY:
  // CHECK-NEXT:  assign f = d & e;
  // CHECK-NEXT:  assign f = d ? d : e;
  // CHECK-NEXT: endmodule
  rtl.module @AAA(%d: i1 {rtl.direction = "in"}, 
                %e: i1 {rtl.direction = "in"}, 
                %f: i1 {rtl.direction = "out"}){}

  rtl.module @AB(%w: i1 {rtl.direction = "in"}, 
                 %x: i1 {rtl.direction = "in"}, 
                 %y: i1 {rtl.direction = "out"},
                 %z: i1 {rtl.direction = "out"}) {

    %w1 = rtl.wire : i1
    %w2 = rtl.wire : i1

    rtl.instance "a1" @AAA(%w, %w1, %w2) : i1, i1, i1
    rtl.instance "b1" @B(%w2, %w1, %y) : i1, i1, i1

    rtl.connect %z, %x : i1
  }
  //CHECK-LABEL: module AB(
  //CHECK-NEXT:   input  w, x,
  //CHECK-NEXT:   output y, z);
  //CHECK-EMPTY: 
  //CHECK-NEXT:   wire w1;
  //CHECK-NEXT:   wire w2;
  //CHECK-EMPTY: 
  //CHECK-NEXT: A a1 (
  //CHECK-NEXT:     .d (w),
  //CHECK-NEXT:     .e (w1),
  //CHECK-NEXT:     .f (w2)
  //CHECK-NEXT:   )
  //CHECK-NEXT: B b1 (
  //CHECK-NEXT:     .a (w2),
  //CHECK-NEXT:     .b (w1),
  //CHECK-NEXT:     .c (y)
  //CHECK-NEXT:   )
  //CHECK-NEXT:   assign z = x;
  //CHECK-NEXT: endmodule

  rtl.module @shl(%a: i1 {rtl.direction = "in"},
                  %b: i1 {rtl.direction = "out"}) {
    %0 = rtl.shl %a, %a : i1
    rtl.connect %b, %0 : i1
  }

  // CHECK-LABEL:  module shl(
  // CHECK-NEXT:   input  a,
  // CHECK-NEXT:   output b);
  // CHECK-EMPTY:
  // CHECK-NEXT:   assign b = a << a;
  // CHECK-NEXT: endmodule

}
