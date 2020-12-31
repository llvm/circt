// RUN: circt-translate %s -emit-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

module {
  rtl.externmodule @E(%a: i1 {rtl.direction = "in"}, 
                %b: i1 {rtl.direction = "out"}, 
                %c: i1 {rtl.direction = "out"})

  // CHECK-LABEL: // external module E

  rtl.module @TESTSIMPLE(%a: i4, %b: i4, %cond: i1) -> (
    %r0: i4, %r2: i4, %r4: i4, %r6: i4,
    %r7: i4, %r8: i4, %r9: i4, %r10: i4,
    %r11: i4, %r12: i4, %r13: i4, %r14: i4,
    %r15: i4, %r16: i1,
    %r17: i1, %r18: i1, %r19: i1, %r20: i1,
    %r21: i1, %r22: i1, %r23: i1, %r24: i1,
    %r25: i1, %r26: i1, %r27: i1, %r28: i1,
    %r29: i12, %r30: i2, %r31: i9, %r32: i9, %r33: i4, %r34: i4
    ) {
    
    %0 = rtl.add %a, %b : i4
    %2 = rtl.sub %a, %b : i4
    %4 = rtl.mul %a, %b : i4
    %6 = rtl.divu %a, %b : i4
    %7 = rtl.divs %a, %b : i4
    %8 = rtl.modu %a, %b : i4
    %9 = rtl.mods %a, %b : i4
    %10 = rtl.shl %a, %b : i4
    %11 = rtl.shru %a, %b : i4
    %12 = rtl.shrs %a, %b : i4
    %13 = rtl.or %a, %b : i4
    %14 = rtl.and %a, %b : i4
    %15 = rtl.xor %a, %b : i4
    %16 = rtl.icmp "eq" %a, %b : i4
    %17 = rtl.icmp "ne" %a, %b : i4
    %18 = rtl.icmp "slt" %a, %b : i4
    %19 = rtl.icmp "sle" %a, %b : i4
    %20 = rtl.icmp "sgt" %a, %b : i4
    %21 = rtl.icmp "sge" %a, %b : i4
    %22 = rtl.icmp "ult" %a, %b : i4
    %23 = rtl.icmp "ule" %a, %b : i4
    %24 = rtl.icmp "ugt" %a, %b : i4
    %25 = rtl.icmp "uge" %a, %b : i4
    %26 = rtl.andr %a : i4
    %27 = rtl.orr %a : i4
    %28 = rtl.xorr %a : i4
    %29 = rtl.concat %a, %a, %b : (i4, i4, i4) -> i12
    %30 = rtl.extract %a from 1 : (i4) -> i2
    %31 = rtl.sext %a : (i4) -> i9
    %32 = rtl.zext %a : (i4) -> i9
    %33 = rtl.mux %cond, %a, %b : i4

    %allone = rtl.constant (15 : i4) : i4
    %34 = rtl.xor %a, %allone : i4

    rtl.output %0, %2, %4, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34:
     i4,i4, i4,i4,i4,i4,i4, i4,i4,i4,i4,i4,
     i4,i1,i1,i1,i1, i1,i1,i1,i1,i1, i1,i1,i1,i1,
     i12, i2,i9,i9,i4, i4
  }
  // CHECK-LABEL: module TESTSIMPLE(
  // CHECK-NEXT:   input  [3:0]  a, b
  // CHECK-NEXT:   input         cond,
  // CHECK-NEXT:   output [3:0]  r0, r2, r4, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
  // CHECK-NEXT:   output        r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28
  // CHECK-NEXT:   output [11:0] r29,
  // CHECK-NEXT:   output [1:0]  r30,
  // CHECK-NEXT:   output [8:0]  r31, r32,
  // CHECK-NEXT:   output [3:0]  r33, r34);
  // CHECK-EMPTY:
  // CHECK-NEXT:   assign r0 = a + b;
  // CHECK-NEXT:   assign r2 = a - b;
  // CHECK-NEXT:   assign r4 = a * b;
  // CHECK-NEXT:   assign r6 = a / b;
  // CHECK-NEXT:   assign r7 = $signed(a) / $signed(b);
  // CHECK-NEXT:   assign r8 = a % b;
  // CHECK-NEXT:   assign r9 = $signed(a) % $signed(b);
  // CHECK-NEXT:   assign r10 = a << b;
  // CHECK-NEXT:   assign r11 = a >> b;
  // CHECK-NEXT:   assign r12 = $signed(a) >>> $signed(b);
  // CHECK-NEXT:   assign r13 = a | b;
  // CHECK-NEXT:   assign r14 = a & b;
  // CHECK-NEXT:   assign r15 = a ^ b;
  // CHECK-NEXT:   assign r16 = a == b;
  // CHECK-NEXT:   assign r17 = a != b;
  // CHECK-NEXT:   assign r18 = $signed(a) < $signed(b);
  // CHECK-NEXT:   assign r19 = $signed(a) <= $signed(b);
  // CHECK-NEXT:   assign r20 = $signed(a) > $signed(b);
  // CHECK-NEXT:   assign r21 = $signed(a) >= $signed(b);
  // CHECK-NEXT:   assign r22 = a < b;
  // CHECK-NEXT:   assign r23 = a <= b;
  // CHECK-NEXT:   assign r24 = a > b;
  // CHECK-NEXT:   assign r25 = a >= b;
  // CHECK-NEXT:   assign r26 = &a;
  // CHECK-NEXT:   assign r27 = |a;
  // CHECK-NEXT:   assign r28 = ^a;
  // CHECK-NEXT:   assign r29 = {a, a, b};
  // CHECK-NEXT:   assign r30 = a[2:1]; 
  // CHECK-NEXT:   assign r31 = {{[{}][{}]}}5{a[3]}}, a};
  // CHECK-NEXT:   assign r32 = {{[{}][{}]}}5'd0}, a};
  // CHECK-NEXT:   assign r33 = cond ? a : b;
  // CHECK-NEXT:   assign r34 = ~a;
  // CHECK-NEXT: endmodule

  rtl.module @B(%a: i1) -> (%b: i1, %c: i1) {
    %0 = rtl.or %a, %a : i1
    %1 = rtl.and %a, %a : i1
    rtl.output %0, %1 : i1, i1
  }
  // CHECK-LABEL: module B(
  // CHECK-NEXT:   input  a,
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
  // CHECK-NEXT:   wire a1_f;
  // CHECK-NEXT:   wire b1_b;
  // CHECK-NEXT:   wire b1_c;
  // CHECK-NEXT:   wire paramd_out;
  // CHECK-NEXT:   wire paramd2_out;
  // CHECK-EMPTY:
  // CHECK-NEXT:   A a1 (
  // CHECK-NEXT:     .d (w),
  // CHECK-NEXT:     .e (b1_b),
  // CHECK-NEXT:     .f (a1_f)
  // CHECK-NEXT:   )
  // CHECK-NEXT:   B b1 (
  // CHECK-NEXT:     .a (a1_f),
  // CHECK-NEXT:     .b (b1_b),
  // CHECK-NEXT:     .c (b1_c)
  // CHECK-NEXT:   )
  // CHECK-NEXT:   FooModule #(.DEFAULT(0), .DEPTH(3.242000e+01), .FORMAT("xyz_timeout=%d\n"), .WIDTH(32)) paramd (
  // CHECK-NEXT:     .a (w),
  // CHECK-NEXT:     .out (paramd_out)
  // CHECK-NEXT:   );
  // CHECK-NEXT:   FooModule #(.DEFAULT(1)) paramd2 (
  // CHECK-NEXT:   .a (i2),
  // CHECK-NEXT:   .out (paramd2_out)
  // CHECK-NEXT:   );
  // CHECK-NEXT:   assign y = b1_c;
  // CHECK-NEXT:   assign z = x;
  // CHECK-NEXT:   assign p = paramd_out;
  // CHECK-NEXT:   assign p2 = paramd2_out;
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
    %aget = rtl.read_inout %a: !rtl.inout<i42>
    rtl.output %aget : i42
  }
  // CHECK-LABEL:  module inout(
  // CHECK-NEXT:     inout  [41:0] a,
  // CHECK-NEXT:     output [41:0] out);
  // CHECK-EMPTY:
  // CHECK-NEXT:     assign out = a;
  // CHECK-NEXT:   endmodule

  // https://github.com/llvm/circt/issues/316
  // FIXME: The MLIR parser doesn't accept an i0 even though it is valid IR,
  // this needs to be fixed upstream.
  //rtl.module @issue316(%inp_0: i0) {
  //  rtl.output
  //}

  // https://github.com/llvm/circt/issues/318
  // This shouldn't generate invalid Verilog
  rtl.module @extract_all(%tmp85: i1) -> (%tmp106: i1) {
    %1 = rtl.extract %tmp85 from 0 : (i1) -> i1
    rtl.output %1 : i1
  }
  // CHECK-LABEL: module extract_all
  // CHECK:  assign tmp106 = tmp85;

  // https://github.com/llvm/circt/issues/320
  rtl.module @literal_extract(%inp_1: i349) -> (%tmp6: i349) {
    %c-58836_i17 = rtl.constant(-58836 : i17) : i17
    %0 = rtl.sext %c-58836_i17 : (i17) -> i349
    rtl.output %0 : i349
  }
  // CHECK-LABEL: module literal_extract
  // CHECK: wire [16:0] _T = 17'h11A2C;
  // CHECK: assign tmp6 = {{[{][{]}}332{_T[16]}}, _T};

  rtl.module @wires(%in4: i4, %in8: i8) -> (%a: i4, %b: i8) {
    // CHECK-LABEL: module wires(
    // CHECK-NEXT:   input  [3:0] in4,
    // CHECK-NEXT:   input  [7:0] in8,
    // CHECK-NEXT:   output [3:0] a,
    // CHECK-NEXT:   output [7:0] b);

    // CHECK-EMPTY:
    %myWire = rtl.wire : !rtl.inout<i4>  // CHECK-NEXT: wire [3:0] myWire;

    // CHECK-NEXT: wire [7:0] myArray1[41:0];
    %myArray1 = rtl.wire : !rtl.inout<array<42 x i8>>
    // CHECK-NEXT: wire [3:0] myWireArray2[41:0][2:0];
    %myWireArray2 = rtl.wire : !rtl.inout<array<3 x array<42 x i4>>>

    // CHECK-EMPTY:
    rtl.connect %myWire, %in4 : i4       // CHECK-NEXT: assign myWire = in4;

    %subscript1 = rtl.arrayindex %myArray1[%in4] : !rtl.inout<array<42 x i8>>, i4
    rtl.connect %subscript1, %in8 : i8   // CHECK-NEXT: assign myArray1[in4] = in8;

    %wireout = rtl.read_inout %myWire : !rtl.inout<i4>

    %subscript2 = rtl.arrayindex %myArray1[%in4] : !rtl.inout<array<42 x i8>>, i4
    %memout = rtl.read_inout %subscript2 : !rtl.inout<i8>

    // CHECK-NEXT: assign a = myWire;
    // CHECK-NEXT: assign b = myArray1[in4];
    rtl.output %wireout, %memout : i4, i8
  }
}

