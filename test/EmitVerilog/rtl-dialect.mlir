// RUN: circt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

module {
  rtl.externmodule @E(%a: i1 {rtl.direction = "in"}, 
                %b: i1 {rtl.direction = "out"}, 
                %c: i1 {rtl.direction = "out"})

  // CHECK-LABEL: // external module E

  rtl.module @TESTSIMPLE(%a: i4, %cond: i1) -> (%r0: i4, %r1: i4, %r2: i4, %r3: i4, %r4: i4, %r5: i4, %r6: i4, %r7: i4, %r8: i4, %r9: i4,
%r10: i1, %r11: i1, %r12: i1, %r13: i1, %r14: i1, %r15: i1, %r16: i1, %r17: i1, %r18: i1, %r19: i1,
%r20: i1, %r21: i1, %r22: i1, %r23: i12, %r24: i2, %r25: i9, %r26: i9, %r27: i4
  ) {
    %0 = rtl.add %a, %a : i4
    %1 = rtl.sub %a, %a : i4
    %2 = rtl.mul %a, %a : i4
    %3 = rtl.div %a, %a : i4
    %4 = rtl.mod %a, %a : i4
    %5 = rtl.shl %a, %a : i4
    %6 = rtl.shr %a, %a : i4
    %7 = rtl.or %a, %a : i4
    %8 = rtl.and %a, %a : i4
    %9 = rtl.xor %a, %a : i4
    %10 = rtl.icmp "eq" %a, %a : i4
    %11 = rtl.icmp "ne" %a, %a : i4
    %12 = rtl.icmp "slt" %a, %a : i4
    %13 = rtl.icmp "sle" %a, %a : i4
    %14 = rtl.icmp "sgt" %a, %a : i4
    %15 = rtl.icmp "sge" %a, %a : i4
    %16 = rtl.icmp "ult" %a, %a : i4
    %17 = rtl.icmp "ule" %a, %a : i4
    %18 = rtl.icmp "ugt" %a, %a : i4
    %19 = rtl.icmp "uge" %a, %a : i4
    %20 = rtl.andr %a : i4
    %21 = rtl.orr %a : i4
    %22 = rtl.xorr %a : i4
    %23 = rtl.concat %a, %a, %a : (i4, i4, i4) -> i12
    %24 = rtl.extract %a from 1 : (i4) -> i2
    %25 = rtl.sext %a : (i4) -> i9
    %26 = rtl.zext %a : (i4) -> i9
    %27 = rtl.mux %cond, %a, %a : i4
    
    rtl.output %0, %1, %2, %3, %4, %5, %6, %7, %8, %9,
    %10, %11, %12, %13, %14, %15, %16, %17, %18, %19,
    %20, %21, %22, %23, %24, %25, %26, %27 :
    i4, i4, i4, i4, i4,
    i4, i4, i4, i4, i4,
    i1, i1, i1, i1, i1,
    i1, i1, i1, i1, i1,
    i1, i1, i1, i12, i2,
    i9, i9, i4
  }
  // CHECK-LABEL: module TESTSIMPLE(
  // CHECK-NEXT:   input  [3:0]  a,
  // CHECK-NEXT:   input         cond,
  // CHECK-NEXT:   output [3:0]  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9,
  // CHECK-NEXT:   output        r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22,
  // CHECK-NEXT:   output [11:0] r23,
  // CHECK-NEXT:   output [1:0]  r24,
  // CHECK-NEXT:   output [8:0]  r25, r26,
  // CHECK-NEXT:   output [3:0]  r27);
  // CHECK-EMPTY:
  // CHECK-NEXT:   assign r0 = a + a;
  // CHECK-NEXT:   assign r1 = a - a;
  // CHECK-NEXT:   assign r2 = a * a;
  // CHECK-NEXT:   assign r3 = a / a;
  // CHECK-NEXT:   assign r4 = a % a;
  // CHECK-NEXT:   assign r5 = a << a;
  // CHECK-NEXT:   assign r6 = a >>> a;
  // CHECK-NEXT:   assign r7 = a | a;
  // CHECK-NEXT:   assign r8 = a & a;
  // CHECK-NEXT:   assign r9 = a ^ a;
  // CHECK-NEXT:   assign r10 = a == a;
  // CHECK-NEXT:   assign r11 = a != a;
  // CHECK-NEXT:   assign r12 = a < a;
  // CHECK-NEXT:   assign r13 = a <= a;
  // CHECK-NEXT:   assign r14 = a < a;
  // CHECK-NEXT:   assign r15 = a <= a;
  // CHECK-NEXT:   assign r16 = a > a;
  // CHECK-NEXT:   assign r17 = a >= a;
  // CHECK-NEXT:   assign r18 = a > a;
  // CHECK-NEXT:   assign r19 = a >= a;
  // CHECK-NEXT:   assign r20 = &a;
  // CHECK-NEXT:   assign r21 = |a;
  // CHECK-NEXT:   assign r22 = ^a;
  // CHECK-NEXT:   assign r23 = {a, a, a};
  // CHECK-NEXT:   assign r24 = a[2:1]; 
  // CHECK-NEXT:   assign r25 = {{[{}][{}]}}5{a[3]}}, a};
  // CHECK-NEXT:   assign r26 = {{[{}][{}]}}5'd0}, a};
  // CHECK-NEXT:   assign r27 = cond ? a : a;
  // CHECK-NEXT: endmodule

  rtl.module @B(%a: i1 { rtl.inout }) -> (%b: i1, %c: i1) {
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

  rtl.module @A(%d: i1, %e: i1) -> (%f: i1) {
    %1 = rtl.mux %d, %d, %e : i1
    rtl.output %1 : i1
  }
  // CHECK-LABEL: module A(
  // CHECK-NEXT:  input  d, e,
  // CHECK-NEXT:  output f);
  // CHECK-EMPTY:
  // CHECK-NEXT:  assign f = d ? d : e;
  // CHECK-NEXT: endmodule

  rtl.module @AAA(%d: i1, %e: i1) -> (%f: i1) {
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
  rtl.externmodule @EXT_W_PARAMS(%a: i1 {rtl.direction = "in"}) -> (%out: i1)
    attributes { verilogName="FooModule" }

  rtl.externmodule @EXT_W_PARAMS2(%a: i2 {rtl.direction = "in"}) -> (%out: i1)
    attributes { verilogName="FooModule" }

  rtl.module @AB(%w: i1, %x: i1, %i2: i2) -> (%y: i1, %z: i1, %p: i1, %p2: i1) {
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


  rtl.module @shl(%a: i1) -> (%b: i1) {
    %0 = rtl.shl %a, %a : i1
    rtl.output %0 : i1
  }
  // CHECK-LABEL:  module shl(
  // CHECK-NEXT:   input  a,
  // CHECK-NEXT:   output b);
  // CHECK-EMPTY:
  // CHECK-NEXT:   assign b = a << a;
  // CHECK-NEXT: endmodule


  rtl.module @inout(%a: !rtl.inout<i42>) -> (%out: i42) {
    %aget = rtl.read_inout %a: (!rtl.inout<i42>) -> i42
    rtl.output %aget : i42
  }
  // CHECK-LABEL:  module inout(
  // CHECK-NEXT:     inout  [41:0] a,
  // CHECK-NEXT:     output [41:0] out);
  // CHECK-EMPTY:
  // CHECK-NEXT:     assign out = a;
  // CHECK-NEXT:   endmodule
}
