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


  /// TODO: Specify parameter declarations.
  rtl.externmodule @EXT_W_PARAMS(%a: i1 {rtl.direction = "in"}) -> (i1 {rtl.name="out"})
    attributes { verilogName="FooModule" }

  rtl.externmodule @EXT_W_PARAMS2(%a: i2 {rtl.direction = "in"}) -> (i1 {rtl.name="out"})
    attributes { verilogName="FooModule" }

  rtl.module @AB(%w: i1, %x: i1, %i2: i2) ->
       (i1 {rtl.name = "y"}, i1 {rtl.name = "z"}, i1 {rtl.name = "p"}, i1 {rtl.name = "p2"}) {
    %w2 = rtl.instance "a1" @AAA(%w, %w1) : (i1, i1) -> (i1)
    %w1, %y = rtl.instance "b1" @B(%w2) : (i1) -> (i1, i1)

    %p = rtl.instance "paramd" @EXT_W_PARAMS(%w) {parameters = {DEFAULT = 0 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}} : (i1) -> i1

    %p2 = rtl.instance "paramd2" @EXT_W_PARAMS2(%i2) {parameters = {DEFAULT = 1 : i64}} : (i2) -> i1

    rtl.output %y, %x, %p, %p2 : i1, i1, i1, i1
  }
  // CHECK-LABEL: module AB(
  // CHECK-NEXT:   input        w, x,
  // CHECK-NEXT:   input  [1:0] i2,
  // CHECK-NEXT:   output       y, z, p, p2);
  // CHECK-EMPTY:
  // CHECK-NEXT:   wire w2;
  // CHECK-NEXT:   wire w1;
  // CHECK-NEXT:   wire y_0;
  // CHECK-NEXT:   wire p_1;
  // CHECK-NEXT:   wire p2_2;
  // CHECK-EMPTY:
  // CHECK-NEXT:   A a1 (
  // CHECK-NEXT:     .d (w),
  // CHECK-NEXT:     .e (w1),
  // CHECK-NEXT:     .f (w2)
  // CHECK-NEXT:   )
  // CHECK-NEXT:   B b1 (
  // CHECK-NEXT:     .a (w2),
  // CHECK-NEXT:     .b (w1),
  // CHECK-NEXT:     .c (y_0)
  // CHECK-NEXT:   )
  // CHECK-NEXT:   FooModule #(.DEFAULT(0), .DEPTH(3.242000e+01), .FORMAT("xyz_timeout=%d\n"), .WIDTH(32)) paramd (
  // CHECK-NEXT:     .a (w),
  // CHECK-NEXT:     .out (p_1)
  // CHECK-NEXT:   );
  // CHECK-NEXT:   FooModule #(.DEFAULT(1)) paramd2 (
  // CHECK-NEXT:   .a (i2),
  // CHECK-NEXT:   .out (p2_2)
  // CHECK-NEXT:   );
  // CHECK-NEXT:   assign y = y_0;
  // CHECK-NEXT:   assign z = x;
  // CHECK-NEXT:   assign p = p_1;
  // CHECK-NEXT:   assign p2 = p2_2;
  // CHECK-NEXT: endmodule



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
