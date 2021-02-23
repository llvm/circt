// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

rtl.module.extern @E(%a: i1 {rtl.direction = "in"}, 
              %b: i1 {rtl.direction = "out"}, 
              %c: i1 {rtl.direction = "out"})

// CHECK-LABEL: // external module E

rtl.module @TESTSIMPLE(%a: i4, %b: i4, %c: i2, %cond: i1,
                        %array2d: !rtl.array<12 x array<10xi4>>,
                        %uarray: !rtl.uarray<16xi8>,
                        %postUArray: i8,
                        %structA: !rtl.struct<foo: i2, bar:i4>) -> (
  %r0: i4, %r2: i4, %r4: i4, %r6: i4,
  %r7: i4, %r8: i4, %r9: i4, %r10: i4,
  %r11: i4, %r12: i4, %r13: i4, %r14: i4,
  %r15: i4, %r16: i1,
  %r17: i1, %r18: i1, %r19: i1, %r20: i1,
  %r21: i1, %r22: i1, %r23: i1, %r24: i1,
  %r25: i1, %r26: i1, %r27: i1, %r28: i1,
  %r29: i12, %r30: i2, %r31: i9, %r33: i4, %r34: i4,
  %r35: !rtl.array<3xi4>, %r36: i12, %r37: i4,
  %r38: !rtl.array<6xi4>, 
  %r40: !rtl.struct<foo: i2, bar:i4>, %r41: !rtl.struct<foo: i2, bar: i4>
  ) {
  
  %0 = comb.add %a, %b : i4
  %2 = comb.sub %a, %b : i4
  %4 = comb.mul %a, %b : i4
  %6 = comb.divu %a, %b : i4
  %7 = comb.divs %a, %b : i4
  %8 = comb.modu %a, %b : i4
  %9 = comb.mods %a, %b : i4
  %10 = comb.shl %a, %b : i4
  %11 = comb.shru %a, %b : i4
  %12 = comb.shrs %a, %b : i4
  %13 = comb.or %a, %b : i4
  %14 = comb.and %a, %b : i4
  %15 = comb.xor %a, %b : i4
  %16 = comb.icmp eq %a, %b : i4
  %17 = comb.icmp ne %a, %b : i4
  %18 = comb.icmp slt %a, %b : i4
  %19 = comb.icmp sle %a, %b : i4
  %20 = comb.icmp sgt %a, %b : i4
  %21 = comb.icmp sge %a, %b : i4
  %22 = comb.icmp ult %a, %b : i4
  %23 = comb.icmp ule %a, %b : i4
  %24 = comb.icmp ugt %a, %b : i4
  %25 = comb.icmp uge %a, %b : i4
  %26 = comb.andr %a : i4
  %zero4 = comb.constant 0 : i4
  %27 = comb.icmp ne %a, %zero4 : i4
  %28 = comb.xorr %a : i4
  %29 = comb.concat %a, %a, %b : (i4, i4, i4) -> i12
  %30 = comb.extract %a from 1 : (i4) -> i2
  %31 = comb.sext %a : (i4) -> i9
  %33 = comb.mux %cond, %a, %b : i4

  %allone = comb.constant 15 : i4
  %34 = comb.xor %a, %allone : i4

  %arrCreated = rtl.array_create %allone, %allone, %allone, %allone, %allone, %allone, %allone, %allone, %allone : (i4)
  %slice1 = rtl.array_slice %arrCreated at %a : (!rtl.array<9xi4>) -> !rtl.array<3xi4>
  %slice2 = rtl.array_slice %arrCreated at %b : (!rtl.array<9xi4>) -> !rtl.array<3xi4>
  %35 = comb.mux %cond, %slice1, %slice2 : !rtl.array<3xi4>

  %ab = comb.add %a, %b : i4
  %subArr = rtl.array_create %allone, %ab, %allone : (i4)
  %38 = rtl.array_concat %subArr, %subArr : !rtl.array<3 x i4>, !rtl.array<3 x i4>

  %elem2d = rtl.array_get %array2d[%a] : !rtl.array<12 x array<10xi4>>
  %37 = rtl.array_get %elem2d[%b] : !rtl.array<10xi4>

  %36 = comb.concat %a, %a, %a : (i4, i4, i4) -> i12

  %39 = rtl.struct_extract %structA["bar"] : !rtl.struct<foo: i2, bar: i4>
  %40 = rtl.struct_inject %structA["bar"], %a : !rtl.struct<foo: i2, bar: i4>
  %41 = rtl.struct_create (%c, %a) : !rtl.struct<foo: i2, bar: i4>
  %42 = rtl.struct_inject %41["bar"], %b : !rtl.struct<foo: i2, bar: i4>

  rtl.output %0, %2, %4, %6, %7, %8, %9, %10, %11, %12, %13, %14,
              %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27,
              %28, %29, %30, %31, %33, %34, %35, %36, %37, %38, %40, %42 :
    i4,i4, i4,i4,i4,i4,i4, i4,i4,i4,i4,i4,
    i4,i1,i1,i1,i1, i1,i1,i1,i1,i1, i1,i1,i1,i1,
   i12, i2, i9, i4, i4, !rtl.array<3xi4>, i12, i4, !rtl.array<6xi4>, 
   !rtl.struct<foo: i2, bar: i4>, !rtl.struct<foo: i2, bar: i4>
}
// CHECK-LABEL: module TESTSIMPLE(
// CHECK-NEXT:   input  [3:0]                                              a, b,
// CHECK-NEXT:   input  [1:0]                                              c,
// CHECK-NEXT:   input                                                     cond,
// CHECK-NEXT:   input  [11:0][9:0][3:0]                                   array2d,
// CHECK-NEXT:   input  [7:0]                                              uarray[15:0], postUArray,
// CHECK-NEXT:   input  struct packed {logic [1:0] foo; logic [3:0] bar; } structA,
// CHECK-NEXT:   output [3:0]                                              r0, r2, r4, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
// CHECK-NEXT:   output                                                    r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27,
// CHECK-NEXT:   output                                                    r28,
// CHECK-NEXT:   output [11:0]                                             r29,
// CHECK-NEXT:   output [1:0]                                              r30,
// CHECK-NEXT:   output [8:0]                                              r31,
// CHECK-NEXT:   output [3:0]                                              r33, r34,
// CHECK-NEXT:   output [2:0][3:0]                                         r35,
// CHECK-NEXT:   output [11:0]                                             r36,
// CHECK-NEXT:   output [3:0]                                              r37,
// CHECK-NEXT:   output [5:0][3:0]                                         r38,
// CHECK-NEXT:   output struct packed {logic [1:0] foo; logic [3:0] bar; } r40, r41);
// CHECK-EMPTY:
// CHECK-NEXT:   wire [8:0][3:0] [[WIRE0:.+]] = {{[{}][{}]}}4'hF}, {4'hF}, {4'hF}, {4'hF}, {4'hF}, {4'hF}, {4'hF}, {4'hF}, {4'hF}};
// CHECK-NEXT:   wire [2:0][3:0] [[WIRE1:.+]] = {{[{}][{}]}}4'hF}, {a + b}, {4'hF}};
// CHECK-NEXT:   wire struct packed {logic [1:0] foo; logic [3:0] bar; } [[WIRE2:.+]] = '{foo: c, bar: a};
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
// CHECK-NEXT:   assign r33 = cond ? a : b;
// CHECK-NEXT:   assign r34 = ~a;
// CHECK-NEXT:   assign r35 = cond ? [[WIRE0]][a+:3] : [[WIRE0]][b+:3];
// CHECK-NEXT:   assign r36 = {3{a}};
// CHECK-NEXT:   assign r37 = array2d[a][b];
// CHECK-NEXT:   assign r38 = {[[WIRE1]], [[WIRE1]]};
// CHECK-NEXT:   assign r40 = '{foo: structA.foo, bar: a};
// CHECK-NEXT:   assign r41 = '{foo: _T_1.foo, bar: b};
// CHECK-NEXT: endmodule


rtl.module @B(%a: i1) -> (%b: i1, %c: i1) {
  %0 = comb.or %a, %a : i1
  %1 = comb.and %a, %a : i1
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
  %1 = comb.mux %d, %d, %e : i1
  rtl.output %1 : i1
}
// CHECK-LABEL: module A(
// CHECK-NEXT:  input  d, e,
// CHECK-NEXT:  output f);
// CHECK-EMPTY:
// CHECK-NEXT:  assign f = d ? d : e;
// CHECK-NEXT: endmodule

rtl.module @AAA(%d: i1, %e: i1) -> (%f: i1) {
  %z = comb.constant 0 : i1
  rtl.output %z : i1
}
// CHECK-LABEL: module AAA(
// CHECK-NEXT:  input  d, e,
// CHECK-NEXT:  output f);
// CHECK-EMPTY:
// CHECK-NEXT:  assign f = 1'h0;
// CHECK-NEXT: endmodule


/// TODO: Specify parameter declarations.
rtl.module.extern @EXT_W_PARAMS(%a: i1 {rtl.direction = "in"}, %b: i0) -> (%out: i1)
  attributes { verilogName="FooModule" }

rtl.module.extern @EXT_W_PARAMS2(%a: i2 {rtl.direction = "in"}) -> (%out: i1)
  attributes { verilogName="FooModule" }

rtl.module @AB(%w: i1, %x: i1, %i2: i2, %i3: i0) -> (%y: i1, %z: i1, %p: i1, %p2: i1) {
  %w2 = rtl.instance "a1" @AAA(%w, %w1) : (i1, i1) -> (i1)
  %w1, %y = rtl.instance "b1" @B(%w2) : (i1) -> (i1, i1)

  %p = rtl.instance "paramd" @EXT_W_PARAMS(%w, %i3) {parameters = {DEFAULT = 14000240888948784983 : i64, DEPTH = 3.242000e+01 : f64, FORMAT = "xyz_timeout=%d\0A", WIDTH = 32 : i8}} : (i1, i0) -> i1

  %p2 = rtl.instance "paramd2" @EXT_W_PARAMS2(%i2) {parameters = {DEFAULT = 1 : i64}} : (i2) -> i1

  rtl.output %y, %x, %p, %p2 : i1, i1, i1, i1
}
// CHECK-LABEL:  module AB(
// CHECK-NEXT:    input                 w, x,
// CHECK-NEXT:    input  [1:0]          i2,
// CHECK-NEXT: // input  /*Zero Width*/ i3,
// CHECK-NEXT:    output                y, z, p, p2);
// CHECK-EMPTY:
// CHECK-NEXT:    wire a1_f;
// CHECK-NEXT:    wire b1_b;
// CHECK-NEXT:    wire b1_c;
// CHECK-NEXT:    wire paramd_out;
// CHECK-NEXT:    wire paramd2_out;
// CHECK-EMPTY:
// CHECK-NEXT:    A a1 (
// CHECK-NEXT:      .d (w),
// CHECK-NEXT:      .e (b1_b),
// CHECK-NEXT:      .f (a1_f)
// CHECK-NEXT:    )
// CHECK-NEXT:    B b1 (
// CHECK-NEXT:      .a (a1_f),
// CHECK-NEXT:      .b (b1_b),
// CHECK-NEXT:      .c (b1_c)
// CHECK-NEXT:    )
// CHECK-NEXT:    FooModule #(
// CHECK-NEXT:      .DEFAULT(64'd14000240888948784983),
// CHECK-NEXT:      .DEPTH(3.242000e+01),
// CHECK-NEXT:      .FORMAT("xyz_timeout=%d\n"),
// CHECK-NEXT:      .WIDTH(8'd32)
// CHECK-NEXT:    ) paramd (
// CHECK-NEXT:      .a   (w),
// CHECK-NEXT:    //.b   (i3),
// CHECK-NEXT:      .out (paramd_out)
// CHECK-NEXT:    );
// CHECK-NEXT:    FooModule #(
// CHECK-NEXT:      .DEFAULT(64'd1)
// CHECK-NEXT:    ) paramd2 (
// CHECK-NEXT:      .a   (i2),
// CHECK-NEXT:      .out (paramd2_out)
// CHECK-NEXT:    );
// CHECK-NEXT:    assign y = b1_c;
// CHECK-NEXT:    assign z = x;
// CHECK-NEXT:    assign p = paramd_out;
// CHECK-NEXT:    assign p2 = paramd2_out;
// CHECK-NEXT:  endmodule


rtl.module @shl(%a: i1) -> (%b: i1) {
  %0 = comb.shl %a, %a : i1
  rtl.output %0 : i1
}
// CHECK-LABEL:  module shl(
// CHECK-NEXT:   input  a,
// CHECK-NEXT:   output b);
// CHECK-EMPTY:
// CHECK-NEXT:   assign b = a << a;
// CHECK-NEXT: endmodule


rtl.module @inout(%a: !rtl.inout<i42>) -> (%out: i42) {
  %aget = sv.read_inout %a: !rtl.inout<i42>
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
  %1 = comb.extract %tmp85 from 0 : (i1) -> i1
  rtl.output %1 : i1
}
// CHECK-LABEL: module extract_all
// CHECK:  assign tmp106 = tmp85;

// https://github.com/llvm/circt/issues/320
rtl.module @literal_extract(%inp_1: i349) -> (%tmp6: i349) {
  %c-58836_i17 = comb.constant -58836 : i17
  %0 = comb.sext %c-58836_i17 : (i17) -> i349
  rtl.output %0 : i349
}
// CHECK-LABEL: module literal_extract
// CHECK: wire [16:0] _T = 17'h11A2C;
// CHECK: assign tmp6 = {{[{][{]}}332{_T[16]}}, _T};

rtl.module @wires(%in4: i4, %in8: i8) -> (%a: i4, %b: i8, %c: i8) {
  // CHECK-LABEL: module wires(
  // CHECK-NEXT:   input  [3:0] in4,
  // CHECK-NEXT:   input  [7:0] in8,
  // CHECK-NEXT:   output [3:0] a,
  // CHECK-NEXT:   output [7:0] b, c);

  // CHECK-EMPTY:

  // Wires.
  // CHECK-NEXT: wire [3:0]            myWire;
  %myWire = sv.wire : !rtl.inout<i4>

  // Packed arrays.

  // CHECK-NEXT: wire [41:0][7:0]      myArray1;
  %myArray1 = sv.wire : !rtl.inout<array<42 x i8>>
  // CHECK-NEXT: wire [2:0][41:0][3:0] myWireArray2;
  %myWireArray2 = sv.wire : !rtl.inout<array<3 x array<42 x i4>>>

  // Unpacked arrays, and unpacked arrays of packed arrays.

  // CHECK-NEXT: wire [7:0]            myUArray1[41:0];
  %myUArray1 = sv.wire : !rtl.inout<uarray<42 x i8>>

  // CHECK-NEXT: wire [41:0][3:0]      myWireUArray2[2:0];
  %myWireUArray2 = sv.wire : !rtl.inout<uarray<3 x array<42 x i4>>>

  // CHECK-EMPTY:

  // Wires.

  // CHECK-NEXT: assign myWire = in4;
  sv.connect %myWire, %in4 : i4
  %wireout = sv.read_inout %myWire : !rtl.inout<i4>

  // Packed arrays.

  %subscript = sv.array_index_inout %myArray1[%in4] : !rtl.inout<array<42 x i8>>, i4
  // CHECK-NEXT: assign myArray1[in4] = in8;
  sv.connect %subscript, %in8 : i8

  %memout1 = sv.read_inout %subscript : !rtl.inout<i8>

    // Unpacked arrays, and unpacked arrays of packed arrays.
  %subscriptu = sv.array_index_inout %myUArray1[%in4] : !rtl.inout<uarray<42 x i8>>, i4
  // CHECK-NEXT: assign myUArray1[in4] = in8;
  sv.connect %subscriptu, %in8 : i8

  %memout2 = sv.read_inout %subscriptu : !rtl.inout<i8>

  // CHECK-NEXT: assign a = myWire;
  // CHECK-NEXT: assign b = myArray1[in4];
  // CHECK-NEXT: assign c = myUArray1[in4];
  rtl.output %wireout, %memout1, %memout2 : i4, i8, i8
}

// CHECK-LABEL: module merge
rtl.module @merge(%in1: i4, %in2: i4, %in3: i4, %in4: i4) -> (%x: i4) {
  // CHECK: wire [3:0] _T;
  // CHECK: assign _T = in1 + in2;
  %a = comb.add %in1, %in2 : i4

  // CHECK-NEXT: assign _T = in2;
  // CHECK-NEXT: assign _T = in3;
  %b = comb.merge %a, %in2, %in3 : i4

  // CHECK: assign x = _T + in4 + in4;
  %c = comb.add %b, %in4, %in4 : i4
  rtl.output %c : i4
}

// CHECK-LABEL: module signs
rtl.module @signs(%in1: i4, %in2: i4, %in3: i4, %in4: i4)  {
  %awire = sv.wire : !rtl.inout<i4>
  // CHECK: wire [3:0] awire;

  // CHECK: assign awire = $unsigned($signed(in1) / $signed(in2)) /
  // CHECK:                $unsigned($signed(in3) / $signed(in4));
  %a1 = comb.divs %in1, %in2: i4
  %a2 = comb.divs %in3, %in4: i4
  %a3 = comb.divu %a1, %a2: i4
  sv.connect %awire, %a3: i4

  // CHECK: assign awire = $unsigned(
  %b1a = comb.divs %in1, %in2: i4
  %b1b = comb.divs %in1, %in2: i4
  %b1c = comb.divs %in1, %in2: i4
  %b1d = comb.divs %in1, %in2: i4
  %b2 = comb.add %b1a, %b1b: i4
  %b3 = comb.mul %b1c, %b1d: i4
  %b4 = comb.divu %b2, %b3: i4
  sv.connect %awire, %b4: i4

  // https://github.com/llvm/circt/issues/369
  // CHECK: assign awire = 4'sh5 / -4'sh3;
  %c5_i4 = comb.constant 5 : i4
  %c-3_i4 = comb.constant -3 : i4
  %divs = comb.divs %c5_i4, %c-3_i4 : i4
  sv.connect %awire, %divs: i4

  rtl.output
}


// CHECK-LABEL: module casts(
// CHECK-NEXT: input  [6:0]      in1,
// CHECK-NEXT: input  [7:0][3:0] in2,
// CHECK-NEXT: output [6:0]      r1,
// CHECK-NEXT: output [31:0]     r2);
rtl.module @casts(%in1: i7, %in2: !rtl.array<8xi4>) -> (%r1: !rtl.array<7xi1>, %r2: i32) {
  // CHECK-EMPTY:
  %r1 = comb.bitcast %in1 : (i7) -> !rtl.array<7xi1>
  %r2 = comb.bitcast %in2 : (!rtl.array<8xi4>) -> i32

  // CHECK-NEXT: wire [31:0] {{.+}} = /*cast(bit[31:0])*/in2;
  // CHECK-NEXT: assign r1 = in1;
  rtl.output %r1, %r2 : !rtl.array<7xi1>, i32
}

// CHECK-LABEL: module TestZero(
// CHECK-NEXT:      input  [3:0]               a,
// CHECK-NEXT:   // input  /*Zero Width*/      zeroBit,
// CHECK-NEXT:   // input  [2:0]/*Zero Width*/ arrZero,
// CHECK-NEXT:      output [3:0]               r0,
// CHECK-NEXT:   // output /*Zero Width*/      rZero,
// CHECK-NEXT:   // output [2:0]/*Zero Width*/ arrZero_0
// CHECK-NEXT:    );
// CHECK-EMPTY:
rtl.module @TestZero(%a: i4, %zeroBit: i0, %arrZero: !rtl.array<3xi0>)
  -> (%r0: i4, %rZero: i0, %arrZero: !rtl.array<3xi0>) {

  %b = comb.add %a, %a : i4
  rtl.output %b, %zeroBit, %arrZero : i4, i0, !rtl.array<3xi0>

  // CHECK-NEXT:   assign r0 = a + a;
  // CHECK-NEXT:   // Zero width: assign rZero = zeroBit;
  // CHECK-NEXT:   // Zero width: assign arrZero_0 = arrZero;
  // CHECK-NEXT: endmodule
}

// CHECK-LABEL: TestZeroInstance
rtl.module @TestZeroInstance(%aa: i4, %azeroBit: i0, %aarrZero: !rtl.array<3xi0>)
  -> (%r0: i4, %rZero: i0, %arrZero: !rtl.array<3xi0>) {

  // CHECK: TestZero iii (	// {{.*}}rtl-dialect.mlir:{{.*}}:19
  // CHECK:   .a       (aa),
  // CHECK: //.zeroBit (azeroBit),
  // CHECK: //.arrZero (aarrZero),
  // CHECK:   .r0      (iii_r0)
  // CHECK: //.rZero   (iii_rZero)
  // CHECK: //.arrZero (iii_arrZero)
  // CHECK: );

  %o1, %o2, %o3 = rtl.instance "iii" @TestZero(%aa, %azeroBit, %aarrZero)
    : (i4, i0, !rtl.array<3xi0>) -> (i4, i0, !rtl.array<3xi0>)

  // CHECK: assign r0 = iii_r0;
  // CHECK: // Zero width: assign rZero = iii_rZero;
  // CHECK: // Zero width: assign arrZero = iii_arrZero;
  rtl.output %o1, %o2, %o3 : i4, i0, !rtl.array<3xi0>
}

// CHECK-LABEL: TestDupInstanceName
rtl.module @TestDupInstanceName(%a: i1) {
  // CHECK: B name (
  %w1, %y1 = rtl.instance "name" @B(%a) : (i1) -> (i1, i1)

  // CHECK: B name_0 (
  %w2, %y2 = rtl.instance "name" @B(%a) : (i1) -> (i1, i1)
}

// CHECK-LABEL: TestEmptyInstanceName
rtl.module @TestEmptyInstanceName(%a: i1) {
  // CHECK: B _T (
  %w1, %y1 = rtl.instance "" @B(%a) : (i1) -> (i1, i1)

  // CHECK: B _T_0 (
  %w2, %y2 = rtl.instance "" @B(%a) : (i1) -> (i1, i1)
}

// CHECK-LABEL: TestInstanceNameValueConflict
rtl.module @TestInstanceNameValueConflict(%a: i1) {
  // CHECK: wire name
  %name = sv.wire : !rtl.inout<i1>

  // CHECK: B name_0 (
  %w, %y = rtl.instance "name" @B(%a) : (i1) -> (i1, i1)
}

// https://github.com/llvm/circt/issues/525
rtl.module @issue525(%struct: i2, %else: i2) -> (%casex: i2) {
  %2 = comb.add %struct, %else : i2
  rtl.output %2 : i2
}
// CHECK-LABEL: module issue525(
// CHECK-NEXT: input  [1:0] struct_0, else_1,
// CHECK-NEXT: output [1:0] casex_2);
// CHECK: assign casex_2 = struct_0 + else_1;


// https://github.com/llvm/circt/issues/438
// CHECK-LABEL: module cyclic
rtl.module @cyclic(%a: i1) -> (i1 {rtl.name = "b"}) {
  // CHECK: wire _T_0;

  // CHECK: wire _T = _T_0 + _T_0;
  %1 = comb.add %0, %0 : i1
  // CHECK: assign _T_0 = a << a;
  %0 = comb.shl %a, %a : i1
  // CHECK: assign b = _T - _T;
  %2 = comb.sub %1, %1 : i1
  rtl.output %2 : i1
}
