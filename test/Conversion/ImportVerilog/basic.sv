// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @Empty {
// CHECK:       }
module Empty;
  ; // empty member
endmodule

// CHECK-LABEL: moore.module @NestedA {
// CHECK:         moore.instance "NestedB" @NestedB
// CHECK:       }
// CHECK-LABEL: moore.module @NestedB {
// CHECK:         moore.instance "NestedC" @NestedC
// CHECK:       }
// CHECK-LABEL: moore.module @NestedC {
// CHECK:       }
module NestedA;
  module NestedB;
    module NestedC;
    endmodule
  endmodule
endmodule

// CHECK-LABEL: moore.module @Child {
// CHECK:       }
module Child;
endmodule

// CHECK-LABEL: moore.module @Parent
// CHECK:         moore.instance "child" @Child
// CHECK:       }
module Parent;
  Child child();
endmodule

// CHECK-LABEL: moore.module @Basic
module Basic;
  // CHECK: %v0 = moore.variable : l1
  // CHECK: %v1 = moore.variable : i32
  // CHECK: %v2 = moore.variable %v1 : i32
  var v0;
  int v1;
  int v2 = v1;

  // CHECK: %w0 = moore.net wire : l1
  // CHECK: %w1 = moore.net wire %w0 : l1
  wire w0;
  wire w1 = w0;
  // CHECK: %w2 = moore.net uwire %w0 : l1
  uwire w2 = w0;
  // CHECK: %w3 = moore.net tri %w0 : l1
  tri w3 = w0;
  // CHECK: %w4 = moore.net triand %w0 : l1
  triand w4 = w0;
  // CHECK: %w5 = moore.net trior %w0 : l1
  trior w5 = w0;
  // CHECK: %w6 = moore.net wand %w0 : l1
  wand w6 = w0;
  // CHECK: %w7 = moore.net wor %w0 : l1
  wor w7 = w0;
  // CHECK: %w8 = moore.net trireg %w0 : l1
  trireg w8 = w0;
  // CHECK: %w9 = moore.net tri0 %w0 : l1
  tri0 w9 = w0;
  // CHECK: %w10 = moore.net tri1 %w0 : l1
  tri1 w10 = w0;
  // CHECK: %w11 = moore.net supply0 : l1
  supply0 w11;
  // CHECK: %w12 = moore.net supply1 : l1
  supply1 w12;

  // CHECK: %b1 = moore.variable : i1
  // CHECK: %b2 = moore.variable %b1 : i1
  bit [0:0] b1;
  bit b2 = b1;

  // CHECK: moore.procedure initial {
  // CHECK: }
  initial;

  // CHECK: moore.procedure final {
  // CHECK: }
  final begin end

  // CHECK: moore.procedure always {
  // CHECK:   %x = moore.variable
  // CHECK:   %y = moore.variable
  // CHECK: }
  always begin
    int x;
    begin
      int y;
    end
  end

  // CHECK: moore.procedure always_comb {
  // CHECK: }
  always_comb begin end

  // CHECK: moore.procedure always_latch {
  // CHECK: }
  always_latch begin end

  // CHECK: moore.procedure always_ff {
  // CHECK: }
  always_ff @* begin end

  // CHECK: moore.assign %v1, %v2 : i32
  assign v1 = v2;
endmodule

// CHECK-LABEL: moore.module @Statements
module Statements;
  bit x, y, z;
  int i;
  initial begin
    // CHECK: %a = moore.variable  : i32
    automatic int a;
    // CHECK moore.blocking_assign %i, %a : i32
    i = a;
    
    //===------------------------------------------------------------------===//
    // Conditional statements

    // CHECK: [[COND:%.+]] = moore.conversion %x : !moore.i1 -> i1
    // CHECK: scf.if [[COND]] {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK: }
    if (x) x = y;

    // CHECK: [[COND0:%.+]] = moore.and %x, %y
    // CHECK: [[COND1:%.+]] = moore.conversion [[COND0]] : !moore.i1 -> i1
    // CHECK: scf.if [[COND1]] {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK: }
    if (x &&& y) x = y;

    // CHECK: [[COND:%.+]] = moore.conversion %x : !moore.i1 -> i1
    // CHECK: scf.if [[COND]] {
    // CHECK:   moore.blocking_assign %x, %z
    // CHECK: } else {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK: }
    if (x) x = z; else x = y;

    // CHECK: [[COND:%.+]] = moore.conversion %x : !moore.i1 -> i1
    // CHECK: scf.if [[COND]] {
    // CHECK:   moore.blocking_assign %x, %x
    // CHECK: } else {
    // CHECK:   [[COND:%.+]] = moore.conversion %y : !moore.i1 -> i1
    // CHECK:   scf.if [[COND]] {
    // CHECK:     moore.blocking_assign %x, %y
    // CHECK:   } else {
    // CHECK:     moore.blocking_assign %x, %z
    // CHECK:   }
    // CHECK: }
    if (x) begin
      x = x;
    end else if (y) begin
      x = y;
    end else begin
      x = z;
    end

    //===------------------------------------------------------------------===//
    // Case statements

    // CHECK: [[TMP1:%.+]] = moore.eq %x, %x : i1 -> i1
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP2]] {
    // CHECK:   moore.blocking_assign %x, %x : i1
    // CHECK: }
    // CHECK: [[TMP3:%.+]] = moore.eq %x, %x : i1 -> i1
    // CHECK: [[TMP4:%.+]] = moore.eq %x, %y : i1 -> i1
    // CHECK: [[TMP5:%.+]] = moore.or [[TMP3]], [[TMP4]] : i1
    // CHECK: [[TMP6:%.+]] = moore.conversion [[TMP5]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP6]] {
    // CHECK:   moore.blocking_assign %x, %y : i1
    // CHECK: }
    case (x)
      x: x = x;
      x, y: x = y;
    endcase

    // CHECK: [[TMP1:%.+]] = moore.eq %x, %x : i1 -> i1
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP2]] {
    // CHECK:   moore.blocking_assign %x, %x : i1
    // CHECK: }
    // CHECK: [[TMP3:%.+]] = moore.eq %x, %x : i1 -> i1
    // CHECK: [[TMP4:%.+]] = moore.eq %x, %y : i1 -> i1
    // CHECK: [[TMP5:%.+]] = moore.or [[TMP3]], [[TMP4]] : i1
    // CHECK: [[TMP6:%.+]] = moore.conversion [[TMP5]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP6]] {
    // CHECK:   moore.blocking_assign %x, %y : i1
    // CHECK: }
    // CHECK: [[TMP7:%.+]] = moore.eq %x, %z : i1 -> i1
    // CHECK: [[TMP8:%.+]] = moore.conversion [[TMP7]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP8]] {
    // CHECK:   moore.blocking_assign %x, %z : i1
    // CHECK: }
    // CHECK: [[TMP9:%.+]] = moore.or [[TMP5]], [[TMP7]] : i1
    // CHECK: [[TMP10:%.+]] = moore.or [[TMP1]], [[TMP9]] : i1
    // CHECK: [[TMP11:%.+]] = moore.not [[TMP10]] : i1
    // CHECK: [[TMP12:%.+]] = moore.conversion [[TMP11]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP12]] {
    // CHECK:   moore.blocking_assign %x, %x : i1
    // CHECK: }
    case (x)
      x: x = x;
      x, y: x = y;
      z: x = z;
      default x = x;
    endcase

    //===------------------------------------------------------------------===//
    // Loop statements

    // CHECK: moore.blocking_assign %y, %x
    // CHECK: scf.while : () -> () {
    // CHECK:   [[COND:%.+]] = moore.conversion %x : !moore.i1 -> i1
    // CHECK:   scf.condition([[COND]])
    // CHECK: } do {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK:   moore.blocking_assign %x, %z
    // CHECK:   scf.yield
    // CHECK: }
    for (y = x; x; x = z) x = y;

    // CHECK: scf.while (%arg0 = %i) : (!moore.i32) -> !moore.i32 {
    // CHECK:   [[TMP0:%.+]] = moore.bool_cast %arg0 : i32 -> i1
    // CHECK:   [[TMP1:%.+]] = moore.conversion [[TMP0]] : !moore.i1 -> i1
    // CHECK:   scf.condition([[TMP1]]) %arg0 : !moore.i32
    // CHECK: } do {
    // CHECK: ^bb0(%arg0: !moore.i32):
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK:   [[TMP0:%.+]] = moore.constant 1 : i32
    // CHECK:   [[TMP1:%.+]] = moore.sub %arg0, [[TMP0]] : i32
    // CHECK:   scf.yield [[TMP1]] : !moore.i32
    // CHECK: }
    repeat (i) x = y;

    // CHECK: scf.while : () -> () {
    // CHECK:   [[COND:%.+]] = moore.conversion %x : !moore.i1 -> i1
    // CHECK:   scf.condition([[COND]])
    // CHECK: } do {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK:   scf.yield
    // CHECK: }
    while (x) x = y;

    // CHECK: scf.while : () -> () {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK:   [[COND:%.+]] = moore.conversion %x : !moore.i1 -> i1
    // CHECK:   scf.condition([[COND]])
    // CHECK: } do {
    // CHECK:   scf.yield
    // CHECK: }
    do x = y; while (x);

    // CHECK: scf.while : () -> () {
    // CHECK:   %true = hw.constant true
    // CHECK:   scf.condition(%true)
    // CHECK: } do {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK:   scf.yield
    // CHECK: }
    forever x = y;

    //===------------------------------------------------------------------===//
    // Assignments

    // CHECK: moore.blocking_assign %x, %y : i1
    x = y;

    // CHECK: moore.blocking_assign %y, %z : i1
    // CHECK: moore.blocking_assign %x, %z : i1
    x = (y = z);

    // CHECK: moore.nonblocking_assign %x, %y : i1
    x <= y;
  end
endmodule

// CHECK-LABEL: moore.module @Expressions {
module Expressions;
  // CHECK: %a = moore.variable : i32
  // CHECK: %b = moore.variable : i32
  // CHECK: %c = moore.variable : i32
  int a, b, c;
  // CHECK: %u = moore.variable : i32
  int unsigned u, w;
  // CHECK: %v = moore.variable : array<2 x i4>
  bit [1:0][3:0] v;
  // CHECK: %d = moore.variable : l32
  // CHECK: %e = moore.variable : l32
  // CHECK: %f = moore.variable : l32
  integer d, e, f;
  integer unsigned g, h, k;
  // CHECK: %x = moore.variable : i1
  bit x;
  // CHECK: %y = moore.variable : l1
  logic y;
  // CHECK: %vec_1 = moore.variable : l32
  logic [31:0] vec_1;
  // CHECK: %vec_2 = moore.variable : l32
  logic [0:31] vec_2;
  // CHECK: %arr = moore.variable : uarray<3 x uarray<6 x i4>>
  bit [4:1] arr [1:3][2:7];
  // CHECK: %struct0 = moore.variable : packed<struct<{a: i32, b: i32}>>
  struct packed {
    int a, b;
  } struct0;
  // CHECK: %struct1 = moore.variable : packed<struct<{c: struct<{a: i32, b: i32}>, d: struct<{a: i32, b: i32}>}>>
  struct packed {
    struct packed {
      int a, b;
    } c, d;
  } struct1;

  initial begin
    // CHECK: moore.constant 0 : i32
    c = '0;
    // CHECK: moore.constant -1 : i32
    c = '1;
    // CHECK: moore.constant 42 : i32
    c = 42;
    // CHECK: moore.constant 42 : i19
    c = 19'd42;
    // CHECK: moore.constant 42 : i19
    c = 19'sd42;
    // CHECK: moore.concat %a, %b, %c : (!moore.i32, !moore.i32, !moore.i32) -> i96
    a = {a, b, c};
    // CHECK: moore.concat %d, %e : (!moore.l32, !moore.l32) -> l64
    d = {d, e};
    // CHECK: %[[VAL_1:.*]] = moore.constant false : i1
    // CHECK: %[[VAL_2:.*]] = moore.concat %[[VAL_1]] : (!moore.i1) -> i1
    // CHECK: %[[VAL_3:.*]] = moore.replicate %[[VAL_2]] : i1 -> i32
    a = {32{1'b0}};
    // CHECK: %[[VAL:.*]] = moore.constant 1 : i32
    // CHECK: moore.extract %vec_1 from %[[VAL]] : l32, i32 -> l3
    y = vec_1[3:1];
    // CHECK: %[[VAL:.*]] = moore.constant 2 : i32
    // CHECK: moore.extract %vec_2 from %[[VAL]] : l32, i32 -> l2
    y = vec_2[2:3];
    // CHECK: moore.extract %d from %x : l32, i1 -> l1
    y = d[x];
    // CHECK: moore.extract %a from %x : i32, i1 -> i1
    x = a[x];
    // CHECK: %[[VAL:.*]] = moore.constant 15 : i32
    // CHECK: moore.extract %vec_1 from %[[VAL]] : l32, i32 -> l1
    y = vec_1[15];
    // CHECK: %[[VAL:.*]] = moore.constant 15 : i32
    // CHECK: moore.extract %vec_1 from %[[VAL]] : l32, i32 -> l1
    y = vec_1[15+:1];
    // CHECK: %[[VAL:.*]] = moore.constant 0 : i32
    // CHECK: moore.extract %vec_2 from %[[VAL]] : l32, i32 -> l1
    y = vec_2[0+:1];
    // CHECK: %[[VAL_1:.*]] = moore.constant 1 : i32
    // CHECK: %[[VAL_2:.*]] = moore.mul %[[VAL_1]], %a : i32
    // CHECK: moore.extract %vec_1 from %[[VAL_2]] : l32, i32 -> l1
    c = vec_1[1*a-:1];
    // CHECK: %[[VAL_1:.*]] = moore.constant 3 : i32
    // CHECK: %[[VAL_2:.*]] = moore.extract %arr from %[[VAL_1]] : uarray<3 x uarray<6 x i4>>, i32 -> uarray<6 x i4>
    // CHECK: %[[VAL_3:.*]] = moore.constant 7 : i32
    // CHECK: %[[VAL_4:.*]] = moore.extract %[[VAL_2]] from %[[VAL_3]] : uarray<6 x i4>, i32 -> i4
    // CHECK: %[[VAL_5:.*]] = moore.constant 3 : i32
    // CHECK: moore.extract %[[VAL_4]] from %[[VAL_5]] : i4, i32 -> i2
    c = arr[3][7][4:3];
    // CHECK: moore.extract %vec_1 from %c : l32, i32 -> l1
    y = vec_1[c];


    //===------------------------------------------------------------------===//
    // Unary operators

    // CHECK: moore.blocking_assign %c, %a : i32
    c = +a;
    // CHECK: moore.neg %a : i32
    c = -a;
    // CHECK: [[TMP1:%.+]] = moore.conversion %v : !moore.array<2 x i4> -> !moore.i32
    // CHECK: [[TMP2:%.+]] = moore.neg [[TMP1]] : i32
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.i32 -> !moore.i32
    c = -v;
    // CHECK: moore.not %a : i32
    c = ~a;
    // CHECK: moore.reduce_and %a : i32 -> i1
    x = &a;
    // CHECK: moore.reduce_and %d : l32 -> l1
    y = &d;
    // CHECK: moore.reduce_or %a : i32 -> i1
    x = |a;
    // CHECK: moore.reduce_xor %a : i32 -> i1
    x = ^a;
    // CHECK: [[TMP:%.+]] = moore.reduce_and %a : i32 -> i1
    // CHECK: moore.not [[TMP]] : i1
    x = ~&a;
    // CHECK: [[TMP:%.+]] = moore.reduce_or %a : i32 -> i1
    // CHECK: moore.not [[TMP]] : i1
    x = ~|a;
    // CHECK: [[TMP:%.+]] = moore.reduce_xor %a : i32 -> i1
    // CHECK: moore.not [[TMP]] : i1
    x = ~^a;
    // CHECK: [[TMP:%.+]] = moore.reduce_xor %a : i32 -> i1
    // CHECK: moore.not [[TMP]] : i1
    x = ^~a;
    // CHECK: [[TMP:%.+]] = moore.bool_cast %a : i32 -> i1
    // CHECK: moore.not [[TMP]] : i1
    x = !a;
    // CHECK: [[PRE:%.+]] = moore.read_lvalue %a : i32
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.add [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[PRE]]
    c = a++;
    // CHECK: [[PRE:%.+]] = moore.read_lvalue %a : i32
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.sub [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[PRE]]
    c = a--;
    // CHECK: [[PRE:%.+]] = moore.read_lvalue %a : i32
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.add [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[POST]]
    c = ++a;
    // CHECK: [[PRE:%.+]] = moore.read_lvalue %a : i32
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.sub [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[POST]]
    c = --a;

    //===------------------------------------------------------------------===//
    // Binary operators

    // CHECK: moore.add %a, %b : i32
    c = a + b;
    // CHECK: [[TMP1:%.+]] = moore.conversion %a : !moore.i32 -> !moore.i32
    // CHECK: [[TMP2:%.+]] = moore.conversion %v : !moore.array<2 x i4> -> !moore.i32
    // CHECK: [[TMP3:%.+]] = moore.add [[TMP1]], [[TMP2]] : i32
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.i32 -> !moore.i32
    c = a + v;
    // CHECK: moore.sub %a, %b : i32
    c = a - b;
    // CHECK: moore.mul %a, %b : i32
    c = a * b;
    // CHECK: moore.divu %h, %k : l32
    g = h / k;
    // CHECK: moore.divs %d, %e : l32
    f = d / e;
    // CHECK: moore.modu %h, %k : l32
    g = h % k;
    // CHECK: moore.mods %d, %e : l32
    f = d % e;

    // CHECK: moore.and %a, %b : i32
    c = a & b;
    // CHECK: moore.or %a, %b : i32
    c = a | b;
    // CHECK: moore.xor %a, %b : i32
    c = a ^ b;
    // CHECK: [[TMP:%.+]] = moore.xor %a, %b : i32
    // CHECK: moore.not [[TMP]] : i32
    c = a ~^ b;
    // CHECK: [[TMP:%.+]] = moore.xor %a, %b : i32
    // CHECK: moore.not [[TMP]] : i32
    c = a ^~ b;

    // CHECK: moore.eq %a, %b : i32 -> i1
    x = a == b;
    // CHECK: moore.eq %d, %e : l32 -> l1
    y = d == e;
    // CHECK: moore.ne %a, %b : i32 -> i1
    x = a != b ;
    // CHECK: moore.case_eq %a, %b : i32
    x = a === b;
    // CHECK: moore.case_ne %a, %b : i32
    x = a !== b;
    // CHECK: moore.wildcard_eq %a, %b : i32 -> i1
    x = a ==? b;
    // CHECK: [[TMP:%.+]] = moore.conversion %a : !moore.i32 -> !moore.l32
    // CHECK: moore.wildcard_eq [[TMP]], %d : l32 -> l1
    y = a ==? d;
    // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.i32 -> !moore.l32
    // CHECK: moore.wildcard_eq %d, [[TMP]] : l32 -> l1
    y = d ==? b;
    // CHECK: moore.wildcard_eq %d, %e : l32 -> l1
    y = d ==? e;
    // CHECK: moore.wildcard_ne %a, %b : i32 -> i1
    x = a !=? b;

    // CHECK: moore.uge %u, %w : i32 -> i1
    c = u >= w;
    // CHECK: moore.ugt %u, %w : i32 -> i1
    c = u > w;
    // CHECK: moore.ule %u, %w : i32 -> i1
    c = u <= w;
    // CHECK: moore.ult %u, %w : i32 -> i1
    c = u < w;
    // CHECK: moore.sge %a, %b : i32 -> i1
    c = a >= b;
    // CHECK: moore.sgt %a, %b : i32 -> i1
    c = a > b;
    // CHECK: moore.sle %a, %b : i32 -> i1
    c = a <= b;
    // CHECK: moore.slt %a, %b : i32 -> i1
    c = a < b;

    // CHECK: [[A:%.+]] = moore.bool_cast %a : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast %b : i32 -> i1
    // CHECK: moore.and [[A]], [[B]] : i1
    c = a && b;
    // CHECK: [[A:%.+]] = moore.bool_cast %a : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast %b : i32 -> i1
    // CHECK: moore.or [[A]], [[B]] : i1
    c = a || b;
    // CHECK: [[A:%.+]] = moore.bool_cast %a : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast %b : i32 -> i1
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : i1
    // CHECK: moore.or [[NOT_A]], [[B]] : i1
    c = a -> b;
    // CHECK: [[A:%.+]] = moore.bool_cast %a : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast %b : i32 -> i1
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : i1
    // CHECK: [[NOT_B:%.+]] = moore.not [[B]] : i1
    // CHECK: [[BOTH:%.+]] = moore.and [[A]], [[B]] : i1
    // CHECK: [[NOT_BOTH:%.+]] = moore.and [[NOT_A]], [[NOT_B]] : i1
    // CHECK: moore.or [[BOTH]], [[NOT_BOTH]] : i1
    c = a <-> b;

    // CHECK: moore.shl %a, %b : i32, i32
    c = a << b;
    // CHECK: moore.shr %a, %b : i32, i32
    c = a >> b;
    // CHECK: moore.shl %a, %b : i32, i32
    c = a <<< b;
    // CHECK: moore.ashr %a, %b : i32, i32
    c = a >>> b;
    // CHECK: moore.shr %u, %b : i32, i32
    c = u >>> b;

    // CHECK: moore.wildcard_eq %a, %a : i32 -> i1
    c = a inside { a };

    // CHECK: [[TMP1:%.+]] = moore.wildcard_eq %a, %a : i32 -> i1
    // CHECK: [[TMP2:%.+]] = moore.wildcard_eq %a, %b : i32 -> i1
    // CHECK: moore.or [[TMP1]], [[TMP2]] : i1
    c = a inside { a, b };

    // CHECK: [[TMP1:%.+]] = moore.wildcard_eq %a, %a : i32 -> i1
    // CHECK: [[TMP2:%.+]] = moore.wildcard_eq %a, %b : i32 -> i1
    // CHECK: [[TMP3:%.+]] = moore.wildcard_eq %a, %a : i32 -> i1
    // CHECK: [[TMP4:%.+]] = moore.wildcard_eq %a, %b : i32 -> i1
    // CHECK: [[TMP5:%.+]] = moore.or [[TMP3]], [[TMP4]] : i1
    // CHECK: [[TMP6:%.+]] = moore.or [[TMP2]], [[TMP5]] : i1
    // CHECK: moore.or [[TMP1]], [[TMP6]] : i1
    c = a inside { a, b, a, b };

    // CHECK: [[TMP1:%.+]] = moore.wildcard_eq %a, %a : i32 -> i1
    // CHECK: [[TMP2:%.+]] = moore.wildcard_eq %a, %b : i32 -> i1
    // CHECK: [[TMP3:%.+]] = moore.sge %a, %a : i32 -> i1
    // CHECK: [[TMP4:%.+]] = moore.sle %a, %b : i32 -> i1
    // CHECK: [[TMP5:%.+]] = moore.and [[TMP3]], [[TMP4]] : i1
    // CHECK: [[TMP6:%.+]] = moore.or [[TMP2]], [[TMP5]] : i1
    // CHECK: moore.or [[TMP1]], [[TMP6]] : i1
    c = a inside { a, b, [a:b] };

    //===------------------------------------------------------------------===//
    // Assign operators

    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.add [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a += b;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.sub [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a -= b;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.mul [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a *= b;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %f
    // CHECK: [[TMP2:%.+]] = moore.divs [[TMP1]], %d
    // CHECK: moore.blocking_assign %f, [[TMP2]]
    f /= d;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %g
    // CHECK: [[TMP2:%.+]] = moore.divu [[TMP1]], %h
    // CHECK: moore.blocking_assign %g, [[TMP2]]
    g /= h;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %f
    // CHECK: [[TMP2:%.+]] = moore.mods [[TMP1]], %d
    // CHECK: moore.blocking_assign %f, [[TMP2]]
    f %= d;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %g
    // CHECK: [[TMP2:%.+]] = moore.modu [[TMP1]], %h
    // CHECK: moore.blocking_assign %g, [[TMP2]]
    g %= h;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.and [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a &= b;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.or [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a |= b;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.xor [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a ^= b;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.shl [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a <<= b;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.shl [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a <<<= b;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.shr [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a >>= b;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP2:%.+]] = moore.ashr [[TMP1]], %b
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a >>>= b;

    // CHECK: [[A_ADD:%.+]] = moore.read_lvalue %a
    // CHECK: [[A_MUL:%.+]] = moore.read_lvalue %a
    // CHECK: [[A_DEC:%.+]] = moore.read_lvalue %a
    // CHECK: [[TMP1:%.+]] = moore.constant 1
    // CHECK: [[TMP2:%.+]] = moore.sub [[A_DEC]], [[TMP1]]
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    // CHECK: [[TMP1:%.+]] = moore.mul [[A_MUL]], [[A_DEC]]
    // CHECK: moore.blocking_assign %a, [[TMP1]]
    // CHECK: [[TMP2:%.+]] = moore.add [[A_ADD]], [[TMP1]]
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a += (a *= a--);

    // CHECK: [[TMP:%.+]] = moore.struct_extract %struct0, "a" : packed<struct<{a: i32, b: i32}>> -> i32
    // CHECK: moore.blocking_assign [[TMP]], %a : i32
    struct0.a = a;

    // CHECK: [[TMP:%.+]]  = moore.struct_extract %struct0, "b" : packed<struct<{a: i32, b: i32}>> -> i32
    // CHECK: moore.blocking_assign %b, [[TMP]] : i32
    b = struct0.b;

    // CHECK: [[TMP1:%.+]] = moore.struct_extract %struct1, "c" : packed<struct<{c: struct<{a: i32, b: i32}>, d: struct<{a: i32, b: i32}>}>> -> packed<struct<{a: i32, b: i32}>>
    // CHECK: [[TMP2:%.+]] = moore.struct_extract [[TMP1]], "a" : packed<struct<{a: i32, b: i32}>> -> i32
    // CHECK: moore.blocking_assign [[TMP2]], %a : i32
    struct1.c.a = a;

    // CHECK: [[TMP1:%.+]] = moore.struct_extract %struct1, "d" : packed<struct<{c: struct<{a: i32, b: i32}>, d: struct<{a: i32, b: i32}>}>> -> packed<struct<{a: i32, b: i32}>>
    // CHECK: [[TMP2:%.+]] = moore.struct_extract [[TMP1]], "b" : packed<struct<{a: i32, b: i32}>> -> i32
    // CHECK: moore.blocking_assign %b, [[TMP2]] : i32
    b = struct1.d.b;
  end
endmodule

// CHECK-LABEL: moore.module @Conversion {
module Conversion;
  // Implicit conversion.
  // CHECK: %a = moore.variable
  // CHECK: [[TMP:%.+]] = moore.conversion %a : !moore.i16 -> !moore.i32
  // CHECK: %b = moore.variable [[TMP]]
  shortint a;
  int b = a;

  // Explicit conversion.
  // CHECK: [[TMP1:%.+]] = moore.conversion %a : !moore.i16 -> !moore.i8
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i8 -> !moore.i32
  // CHECK: %c = moore.variable [[TMP2]]
  int c = byte'(a);

  // Sign conversion.
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.i32 -> !moore.i32
  // CHECK: %d1 = moore.variable [[TMP]]
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.i32 -> !moore.i32
  // CHECK: %d2 = moore.variable [[TMP]]
  bit signed [31:0] d1 = signed'(b);
  bit [31:0] d2 = unsigned'(b);

  // Width conversion.
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.i32 -> !moore.i19
  // CHECK: %e = moore.variable [[TMP]]
  bit signed [18:0] e = 19'(b);
endmodule
