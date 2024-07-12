// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @Empty() {
// CHECK:       }
module Empty;
  ; // empty member
endmodule

// CHECK-LABEL: moore.module @NestedA() {
// CHECK:         moore.instance "NestedB" @NestedB
// CHECK:       }
// CHECK-LABEL: moore.module private @NestedB() {
// CHECK:         moore.instance "NestedC" @NestedC
// CHECK:       }
// CHECK-LABEL: moore.module private @NestedC() {
// CHECK:       }
module NestedA;
  module NestedB;
    module NestedC;
    endmodule
  endmodule
endmodule

// CHECK-LABEL: moore.module private @Child() {
// CHECK:       }
module Child;
endmodule

// CHECK-LABEL: moore.module @Parent() {
// CHECK:         moore.instance "child" @Child
// CHECK:       }
module Parent;
  Child child();
endmodule

// CHECK-LABEL: moore.module @Basic
module Basic;
  // CHECK: %v0 = moore.variable : <l1>
  // CHECK: %v1 = moore.variable : <i32>
  // CHECK: [[TMP1:%.+]] = moore.read %v1 : i32
  // CHECK: %v2 = moore.variable [[TMP1]] : <i32>
  var v0;
  int v1;
  int v2 = v1;

  // CHECK: %w0 = moore.net wire : <l1>
  wire w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w1 = moore.net wire [[TMP1]] : <l1>
  wire w1 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w2 = moore.net uwire [[TMP1]] : <l1>
  uwire w2 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w3 = moore.net tri [[TMP1]] : <l1>
  tri w3 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w4 = moore.net triand [[TMP1]] : <l1>
  triand w4 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w5 = moore.net trior [[TMP1]] : <l1>
  trior w5 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w6 = moore.net wand [[TMP1]] : <l1>
  wand w6 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w7 = moore.net wor [[TMP1]] : <l1>
  wor w7 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w8 = moore.net trireg [[TMP1]] : <l1>
  trireg w8 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w9 = moore.net tri0 [[TMP1]] : <l1>
  tri0 w9 = w0;
  // CHECK: [[TMP1:%.+]] = moore.read %w0 : l1
  // CHECK: %w10 = moore.net tri1 [[TMP1]] : <l1>
  tri1 w10 = w0;
  // CHECK: %w11 = moore.net supply0 : <l1>
  supply0 w11;
  // CHECK: %w12 = moore.net supply1 : <l1>
  supply1 w12;

  // CHECK: %b1 = moore.variable : <i1>
  // CHECK: [[TMP1:%.+]] = moore.read %b1 : i1
  // CHECK: %b2 = moore.variable [[TMP1]] : <i1>
  bit [0:0] b1;
  bit b2 = b1;

  // CHECK: [[TMP:%.+]] = moore.constant 1 : l32
  // CHECK: %p1 = moore.named_constant parameter [[TMP]] : l32
  parameter p1 = 1;

  // CHECK: [[TMP:%.+]] = moore.constant 1 : l32
  // CHECK: %p2 = moore.named_constant parameter [[TMP]] : l32
  parameter p2 = p1;

  // CHECK: [[TMP:%.+]] = moore.constant 2 : l32
  // CHECK: %lp1 = moore.named_constant localparam [[TMP]] : l32
  localparam lp1 = 2;

  // CHECK: [[TMP:%.+]] = moore.constant 2 : l32
  // CHECK: %lp2 = moore.named_constant localparam [[TMP]] : l32
  localparam lp2 = lp1;

  // CHECK: [[TMP:%.+]] = moore.constant 3 : l32
  // CHECK: %sp1 = moore.named_constant specparam [[TMP]] : l32
  specparam sp1 = 3;

  // CHECK: [[TMP:%.+]] = moore.constant 3 : l32
  // CHECK: %sp2 = moore.named_constant specparam [[TMP]] : l32
  specparam sp2 = sp1;

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

  // CHECK: [[TMP1:%.+]] = moore.read %v2 : i32
  // CHECK: moore.assign %v1, [[TMP1]] : i32
  assign v1 = v2;
endmodule

// CHECK-LABEL: moore.module @Statements
module Statements;
  bit x, y, z;
  int i;
  initial begin
    // CHECK: %a = moore.variable  : <i32>
    automatic int a;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK moore.blocking_assign %i, [[TMP1]] : i32
    i = a;

    //===------------------------------------------------------------------===//
    // Conditional statements

    // CHECK: [[TMP1:%.+]] = moore.read %x : i1 
    // CHECK: [[COND:%.+]] = moore.conversion [[TMP1]] : !moore.i1 -> i1
    // CHECK: scf.if [[COND]] {
    // CHECK:   [[TMP2:%.+]] = moore.read %y : i1  
    // CHECK:   moore.blocking_assign %x, [[TMP2]] : i1
    // CHECK: }
    if (x) x = y;
    
    // CHECK: [[TMP1:%.+]] = moore.read %x : i1
    // CHECK: [[TMP2:%.+]] = moore.read %y : i1
    // CHECK: [[COND0:%.+]] = moore.and [[TMP1]], [[TMP2]]
    // CHECK: [[COND1:%.+]] = moore.conversion [[COND0]] : !moore.i1 -> i1
    // CHECK: scf.if [[COND1]] {
    // CHECK:   [[TMP3:%.+]] = moore.read %y : i1
    // CHECK:   moore.blocking_assign %x, [[TMP3]]
    // CHECK: }
    if (x &&& y) x = y;
    
    // CHECK: [[TMP1:%.+]] = moore.read %x : i1
    // CHECK: [[COND:%.+]] = moore.conversion [[TMP1]] : !moore.i1 -> i1
    // CHECK: scf.if [[COND]] {
    // CHECK:   [[TMP2:%.+]] = moore.read %z : i1
    // CHECK:   moore.blocking_assign %x, [[TMP2]]
    // CHECK: } else {
    // CHECK:   [[TMP3:%.+]] = moore.read %y : i1
    // CHECK:   moore.blocking_assign %x, [[TMP3]]
    // CHECK: }
    if (x) x = z; else x = y;

    // CHECK: [[TMP1:%.+]] = moore.read %x : i1
    // CHECK: [[COND:%.+]] = moore.conversion [[TMP1]] : !moore.i1 -> i1
    // CHECK: scf.if [[COND]] {
    // CHECK:   [[TMP2:%.+]] = moore.read %x : i1
    // CHECK:   moore.blocking_assign %x, [[TMP2]]
    // CHECK: } else {
    // CHECK:   [[TMP3:%.+]] = moore.read %y : i1
    // CHECK:   [[COND:%.+]] = moore.conversion [[TMP3]] : !moore.i1 -> i1
    // CHECK:   scf.if [[COND]] {
    // CHECK:     [[TMP4:%.+]] = moore.read %y : i1
    // CHECK:     moore.blocking_assign %x, [[TMP4]]
    // CHECK:   } else {
    // CHECK:     [[TMP5:%.+]] = moore.read %z : i1
    // CHECK:     moore.blocking_assign %x, [[TMP5]]
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


    // CHECK: [[TMP1:%.+]] = moore.read %x : i1
    // CHECK: [[TMP2:%.+]] = moore.read %x : i1
    // CHECK: [[TMP3:%.+]] = moore.eq [[TMP1]], [[TMP2]] : i1 -> i1
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP4]] {
    // CHECK:   [[TMP5:%.+]] = moore.read %x : i1
    // CHECK:   moore.blocking_assign %x, [[TMP5]] : i1
    // CHECK: }
    // CHECK: [[TMP6:%.+]] = moore.read %x : i1
    // CHECK: [[TMP7:%.+]] = moore.eq [[TMP1]], [[TMP6]] : i1 -> i1
    // CHECK: [[TMP8:%.+]] = moore.read %y : i1
    // CHECK: [[TMP9:%.+]] = moore.eq [[TMP1]], [[TMP8]] : i1 -> i1
    // CHECK: [[TMP10:%.+]] = moore.or [[TMP7]], [[TMP9]] : i1
    // CHECK: [[TMP11:%.+]] = moore.conversion [[TMP10]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP11]] {
    // CHECK:   [[TMP12:%.+]] = moore.read %y : i1
    // CHECK:   moore.blocking_assign %x, [[TMP12]] : i1
    // CHECK: }
    case (x)
      x: x = x;
      x, y: x = y;
    endcase

    // CHECK: [[TMP1:%.+]] = moore.read %x : i1
    // CHECK: [[TMP2:%.+]] = moore.read %x : i1
    // CHECK: [[TMP3:%.+]] = moore.eq [[TMP1]], [[TMP2]] : i1 -> i1
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP4]] {
    // CHECK:   [[TMP5:%.+]] = moore.read %x : i1
    // CHECK:   moore.blocking_assign %x, [[TMP5]] : i1
    // CHECK: }
    // CHECK: [[TMP6:%.+]] = moore.read %x : i1
    // CHECK: [[TMP7:%.+]] = moore.eq [[TMP1]], [[TMP6]] : i1 -> i1
    // CHECK: [[TMP8:%.+]] = moore.read %y : i1
    // CHECK: [[TMP9:%.+]] = moore.eq [[TMP1]], [[TMP8]] : i1 -> i1
    // CHECK: [[TMP10:%.+]] = moore.or [[TMP7]], [[TMP9]] : i1
    // CHECK: [[TMP11:%.+]] = moore.conversion [[TMP10]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP11]] {
    // CHECK:   [[TMP12:%.+]] = moore.read %y : i1
    // CHECK:   moore.blocking_assign %x, [[TMP12]] : i1
    // CHECK: }
    // CHECK: [[TMP13:%.+]] = moore.read %z : i1
    // CHECK: [[TMP14:%.+]] = moore.eq [[TMP1]], [[TMP13]] : i1 -> i1
    // CHECK: [[TMP15:%.+]] = moore.conversion [[TMP14]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP15]] {
    // CHECK:   [[TMP16:%.+]] = moore.read %z : i1
    // CHECK:   moore.blocking_assign %x, [[TMP16]] : i1
    // CHECK: }
    // CHECK: [[TMP17:%.+]] = moore.or [[TMP10]], [[TMP14]] : i1
    // CHECK: [[TMP18:%.+]] = moore.or [[TMP3]], [[TMP17]] : i1
    // CHECK: [[TMP19:%.+]] = moore.not [[TMP18]] : i1
    // CHECK: [[TMP20:%.+]] = moore.conversion [[TMP19]] : !moore.i1 -> i1
    // CHECK: scf.if [[TMP20]] {
    // CHECK:   [[TMP21:%.+]] = moore.read %x : i1
    // CHECK:   moore.blocking_assign %x, [[TMP21]] : i1
    // CHECK: }
    case (x)
      x: x = x;
      x, y: x = y;
      z: x = z;
      default x = x;
    endcase

    //===------------------------------------------------------------------===//
    // Loop statements

    // CHECK: [[TMP1:%.+]] = moore.read %x : i1
    // CHECK: moore.blocking_assign %y, [[TMP1]] : i1
    // CHECK: scf.while : () -> () {
    // CHECK:   [[TMP2:%.+]] = moore.read %x : i1
    // CHECK:   [[COND:%.+]] = moore.conversion [[TMP2]] : !moore.i1 -> i1
    // CHECK:   scf.condition([[COND]])
    // CHECK: } do {
    // CHECK:   [[TMP3:%.+]] = moore.read %y : i1
    // CHECK:   moore.blocking_assign %x, [[TMP3]] : i1
    // CHECK:   [[TMP4:%.+]] = moore.read %z : i1
    // CHECK:   moore.blocking_assign %x, [[TMP4]] : i1
    // CHECK:   scf.yield
    // CHECK: }
    for (y = x; x; x = z) x = y;
    
    // CHECK: [[TMP1:%.+]] = moore.read %i : i32
    // CHECK: scf.while (%arg0 = [[TMP1]]) : (!moore.i32) -> !moore.i32 {
    // CHECK:   [[TMP2:%.+]] = moore.bool_cast %arg0 : i32 -> i1
    // CHECK:   [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.i1 -> i1
    // CHECK:   scf.condition([[TMP3]]) %arg0 : !moore.i32
    // CHECK: } do {
    // CHECK: ^bb0(%arg0: !moore.i32):
    // CHECK:   [[TMP4:%.+]] = moore.read %y : i1
    // CHECK:   moore.blocking_assign %x, [[TMP4]] : i1
    // CHECK:   [[TMP5:%.+]] = moore.constant 1 : i32
    // CHECK:   [[TMP6:%.+]] = moore.sub %arg0, [[TMP5]] : i32
    // CHECK:   scf.yield [[TMP6]] : !moore.i32
    // CHECK: }
    repeat (i) x = y;

    // CHECK: scf.while : () -> () {
    // CHECK:   [[TMP1:%.+]] = moore.read %x : i1
    // CHECK:   [[COND:%.+]] = moore.conversion [[TMP1]] : !moore.i1 -> i1
    // CHECK:   scf.condition([[COND]])
    // CHECK: } do {
    // CHECK:   [[TMP2:%.+]] = moore.read %y : i1
    // CHECK:   moore.blocking_assign %x, [[TMP2]] : i1
    // CHECK:   scf.yield
    // CHECK: }
    while (x) x = y;

    // CHECK: scf.while : () -> () {
    // CHECK:   [[TMP1:%.+]] = moore.read %y : i1
    // CHECK:   moore.blocking_assign %x, [[TMP1]] : i1
    // CHECK:   [[TMP2:%.+]] = moore.read %x : i1
    // CHECK:   [[COND:%.+]] = moore.conversion [[TMP2]] : !moore.i1 -> i1
    // CHECK:   scf.condition([[COND]])
    // CHECK: } do {
    // CHECK:   scf.yield
    // CHECK: }
    do x = y; while (x);

    // CHECK: scf.while : () -> () {
    // CHECK:   %true = hw.constant true
    // CHECK:   scf.condition(%true)
    // CHECK: } do {
    // CHECK:   [[TMP1:%.+]] = moore.read %y : i1
    // CHECK:   moore.blocking_assign %x, [[TMP1]] : i1
    // CHECK:   scf.yield
    // CHECK: }
    forever x = y;

    //===------------------------------------------------------------------===//
    // Assignments

    // CHECK: [[TMP1:%.+]] = moore.read %y : i1
    // CHECK: moore.blocking_assign %x, [[TMP1]] : i1
    x = y;

    // CHECK: [[TMP1:%.+]] = moore.read %z : i1
    // CHECK: moore.blocking_assign %y, [[TMP1]] : i1
    // CHECK: moore.blocking_assign %x, [[TMP1]] : i1
    x = (y = z);
    
    // CHECK: [[TMP1:%.+]] = moore.read %y : i1
    // CHECK: moore.nonblocking_assign %x, [[TMP1]] : i1
    x <= y;
  end
endmodule

// CHECK-LABEL: moore.module @Expressions
module Expressions;
  // CHECK: %a = moore.variable : <i32>
  // CHECK: %b = moore.variable : <i32>
  // CHECK: %c = moore.variable : <i32>
  int a, b, c;
  // CHECK: %u = moore.variable : <i32>
  int unsigned u, w;
  // CHECK: %v = moore.variable : <array<2 x i4>>
  bit [1:0][3:0] v;
  // CHECK: %d = moore.variable : <l32>
  // CHECK: %e = moore.variable : <l32>
  // CHECK: %f = moore.variable : <l32>
  integer d, e, f;
  integer unsigned g, h, k;
  // CHECK: %x = moore.variable : <i1>
  bit x;
  // CHECK: %y = moore.variable : <l1>
  logic y;
  // CHECK: %vec_1 = moore.variable : <l32>
  logic [31:0] vec_1;
  // CHECK: %vec_2 = moore.variable : <l32>
  logic [0:31] vec_2;
  // CHECK: %arr = moore.variable : <uarray<3 x uarray<6 x i4>>>
  bit [4:1] arr [1:3][2:7];
  // CHECK: %struct0 = moore.variable : <struct<{a: i32, b: i32}>>
  struct packed {
    int a, b;
  } struct0;
  // CHECK: %struct1 = moore.variable : <struct<{c: struct<{a: i32, b: i32}>, d: struct<{a: i32, b: i32}>}>>
  struct packed {
    struct packed {
      int a, b;
    } c, d;
  } struct1;
  // CHECK: %union0 = moore.variable : <union<{a: i32, b: i32}>>
  union packed {
    int a, b;
  } union0;
  // CHECK: %union1 = moore.variable : <union<{c: union<{a: i32, b: i32}>, d: union<{a: i32, b: i32}>}>>
  union packed {
    union packed {
      int a, b;
    } c, d;
  } union1;
  // CHECK: %r1 = moore.variable : <real>
  // CHECK: %r2 = moore.variable : <real>
  real r1,r2;

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
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: [[TMP3:%.+]] = moore.read %c : i32
    // CHECK: moore.concat [[TMP1]], [[TMP2]], [[TMP3]] : (!moore.i32, !moore.i32, !moore.i32) -> i96
    a = {a, b, c};
    // CHECK: [[TMP1:%.+]] = moore.read %d : l32
    // CHECK: [[TMP2:%.+]] = moore.read %e : l32
    // CHECK: moore.concat [[TMP1]], [[TMP2]] : (!moore.l32, !moore.l32) -> l64
    d = {d, e};
    // CHECK: moore.concat_ref %a, %b, %c : (!moore.ref<i32>, !moore.ref<i32>, !moore.ref<i32>) -> <i96>
    {a, b, c} = a;
    // CHECK: moore.concat_ref %d, %e : (!moore.ref<l32>, !moore.ref<l32>) -> <l64>
    {d, e} = d;
    // CHECK: [[TMP1:%.+]] = moore.constant false : i1
    // CHECK: [[TMP2:%.+]] = moore.concat [[TMP1]] : (!moore.i1) -> i1
    // CHECK: moore.replicate [[TMP2]] : i1 -> i32
    a = {32{1'b0}};
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1 : l32
    // CHECK: [[TMP2:%.+]] = moore.constant 1 : i32
    // CHECK: moore.extract [[TMP1]] from [[TMP2]] : l32, i32 -> l3
    y = vec_1[3:1];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_2 : l32
    // CHECK: [[TMP2:%.+]] = moore.constant 2 : i32
    // CHECK: moore.extract [[TMP1]] from [[TMP2]] : l32, i32 -> l2
    y = vec_2[2:3];
    // CHECK: [[TMP1:%.+]] = moore.read %d : l32
    // CHECK: [[TMP2:%.+]] = moore.read %x : i1
    // CHECK: moore.extract [[TMP1]] from [[TMP2]] : l32, i1 -> l1
    y = d[x];
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %x : i1
    // CHECK: moore.extract [[TMP1]] from [[TMP2]] : i32, i1 -> i1
    x = a[x];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1 : l32
    // CHECK: [[TMP2:%.+]] = moore.constant 15 : i32
    // CHECK: moore.extract [[TMP1]] from [[TMP2]] : l32, i32 -> l1
    y = vec_1[15];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1 : l32
    // CHECK: [[TMP2:%.+]] = moore.constant 15 : i32
    // CHECK: moore.extract [[TMP1]] from [[TMP2]] : l32, i32 -> l1
    y = vec_1[15+:1];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_2 : l32
    // CHECK: [[TMP2:%.+]] = moore.constant 0 : i32
    // CHECK: moore.extract [[TMP1]] from [[TMP2]] : l32, i32 -> l1
    y = vec_2[0+:1];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1 : l32
    // CHECK: [[TMP2:%.+]] = moore.constant 1 : i32
    // CHECK: [[TMP3:%.+]] = moore.read %a : i32
    // CHECK: [[TMP4:%.+]] = moore.mul [[TMP2]], [[TMP3]] : i32
    // CHECK: [[TMP5:%.+]] = moore.constant 0 : i32
    // CHECK: [[TMP6:%.+]] = moore.sub [[TMP4]], [[TMP5]] : i32
    // CHECK: moore.extract [[TMP1]] from [[TMP6]] : l32, i32 -> l1
    c = vec_1[1*a-:1];
    // CHECK: [[TMP1:%.+]] = moore.read %arr : uarray<3 x uarray<6 x i4>>
    // CHECK: [[TMP2:%.+]] = moore.constant 3 : i32
    // CHECK: [[TMP3:%.+]] = moore.extract [[TMP1]] from [[TMP2]] : uarray<3 x uarray<6 x i4>>, i32 -> uarray<6 x i4>
    // CHECK: [[TMP4:%.+]] = moore.constant 7 : i32
    // CHECK: [[TMP5:%.+]] = moore.extract [[TMP3]] from [[TMP4]] : uarray<6 x i4>, i32 -> i4
    // CHECK: [[TMP6:%.+]] = moore.constant 3 : i32
    // CHECK: moore.extract [[TMP5]] from [[TMP6]] : i4, i32 -> i2
    c = arr[3][7][4:3];
    // CHECK: [[TMP1:%.+]] = moore.read %vec_1 : l32
    // CHECK: [[TMP2:%.+]] = moore.read %c : i32
    // CHECK: moore.extract [[TMP1]] from [[TMP2]] : l32, i32 -> l1
    y = vec_1[c];

    // CHECK: [[TMP1:%.+]] = moore.constant 1 : i32
    // CHECK: [[TMP2:%.+]] = moore.extract_ref %v from [[TMP1]] : <array<2 x i4>>, i32 -> <i4>
    // CHECK: [[TMP3:%.+]] = moore.constant 3 : i32
    // CHECK: moore.extract_ref [[TMP2]] from [[TMP3]] : <i4>, i32 -> <i1>
    v[1][3] = x;

    // CHECK: [[TMP1:%.+]] = moore.constant 1 : i32
    // CHECK: moore.extract_ref %vec_1 from [[TMP1]] : <l32>, i32 -> <l2>
    vec_1[2:1] = y;

    // CHECK: [[X_READ:%.+]] = moore.read %x : i1
    // CHECK: moore.extract_ref %vec_1 from [[X_READ]] : <l32>, i1 -> <l1>
    vec_1[x] = y;
    
    // CHECK: [[CONST_15:%.+]] = moore.constant 15 : i32
    // CHECK: [[CONST_2:%.+]] = moore.constant 2 : i32
    // CHECK: [[SUB:%.+]] = moore.sub [[CONST_15]], [[CONST_2]] : i32
    // CHECK: moore.extract_ref %vec_1 from [[SUB]] : <l32>, i32 -> <l3>
    vec_1[15-:3] = y;

    //===------------------------------------------------------------------===//
    // Unary operators

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: moore.blocking_assign %c, [[TMP1]] : i32
    c = +a;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: moore.neg [[TMP1]] : i32
    c = -a;
    // CHECK: [[TMP1:%.+]] = moore.read %v : array<2 x i4>
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.array<2 x i4> -> !moore.i32
    // CHECK: [[TMP3:%.+]] = moore.neg [[TMP2]] : i32
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.i32 -> !moore.i32
    c = -v;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: moore.not [[TMP1]] : i32
    c = ~a;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: moore.reduce_and [[TMP1]] : i32 -> i1
    x = &a;
    // CHECK: [[TMP1:%.+]] = moore.read %d : l32
    // CHECK: moore.reduce_and [[TMP1]] : l32 -> l1
    y = &d;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: moore.reduce_or [[TMP1]] : i32 -> i1
    x = |a;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: moore.reduce_xor [[TMP1]] : i32 -> i1
    x = ^a;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.reduce_and [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = ~&a;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.reduce_or [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = ~|a;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.reduce_xor [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = ~^a;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.reduce_xor [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = ^~a;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: moore.not [[TMP2]] : i1
    x = !a;
    // CHECK: [[PRE:%.+]] = moore.read %a : i32
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.add [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[PRE]]
    c = a++;
    // CHECK: [[PRE:%.+]] = moore.read %a : i32
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.sub [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[PRE]]
    c = a--;
    // CHECK: [[PRE:%.+]] = moore.read %a : i32
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.add [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[POST]]
    c = ++a;
    // CHECK: [[PRE:%.+]] = moore.read %a : i32
    // CHECK: [[TMP:%.+]] = moore.constant 1 : i32
    // CHECK: [[POST:%.+]] = moore.sub [[PRE]], [[TMP]] : i32
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[POST]]
    c = --a;

    //===------------------------------------------------------------------===//
    // Binary operators

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.add [[TMP1]], [[TMP2]] : i32
    c = a + b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.i32
    // CHECK: [[TMP3:%.+]] = moore.read %v : array<2 x i4>
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.array<2 x i4> -> !moore.i32
    // CHECK: moore.add [[TMP2]], [[TMP4]] : i32
    c = a + v;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.sub [[TMP1]], [[TMP2]] : i32
    c = a - b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.mul [[TMP1]], [[TMP2]] : i32
    c = a * b;
    // CHECK: [[TMP1:%.+]] = moore.read %h : l32
    // CHECK: [[TMP2:%.+]] = moore.read %k : l32
    // CHECK: moore.divu [[TMP1]], [[TMP2]] : l32
    g = h / k;
    // CHECK: [[TMP1:%.+]] = moore.read %d : l32
    // CHECK: [[TMP2:%.+]] = moore.read %e : l32
    // CHECK: moore.divs [[TMP1]], [[TMP2]] : l32
    f = d / e;
    // CHECK: [[TMP1:%.+]] = moore.read %h : l32
    // CHECK: [[TMP2:%.+]] = moore.read %k : l32
    // CHECK: moore.modu [[TMP1]], [[TMP2]] : l32
    g = h % k;
    // CHECK: [[TMP1:%.+]] = moore.read %d : l32
    // CHECK: [[TMP2:%.+]] = moore.read %e : l32
    // CHECK: moore.mods [[TMP1]], [[TMP2]] : l32
    f = d % e;

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.and [[TMP1]], [[TMP2]] : i32
    c = a & b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.or [[TMP1]], [[TMP2]] : i32
    c = a | b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.xor [[TMP1]], [[TMP2]] : i32
    c = a ^ b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: [[TMP3:%.+]] = moore.xor [[TMP1]], [[TMP2]] : i32
    // CHECK: moore.not [[TMP3]] : i32
    c = a ~^ b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: [[TMP3:%.+]] = moore.xor [[TMP1]], [[TMP2]] : i32
    // CHECK: moore.not [[TMP3]] : i32
    c = a ^~ b;

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.eq [[TMP1]], [[TMP2]] : i32 -> i1
    x = a == b;
    // CHECK: [[TMP1:%.+]] = moore.read %d : l32
    // CHECK: [[TMP2:%.+]] = moore.read %e : l32
    // CHECK: moore.eq [[TMP1]], [[TMP2]] : l32 -> l1
    y = d == e;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.ne [[TMP1]], [[TMP2]] : i32 -> i1
    x = a != b ;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.case_eq [[TMP1]], [[TMP2]] : i32
    x = a === b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.case_ne [[TMP1]], [[TMP2]] : i32
    x = a !== b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    x = a ==? b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.l32
    // CHECK: [[TMP3:%.+]] = moore.read %d : l32
    // CHECK: moore.wildcard_eq [[TMP2]], [[TMP3]] : l32 -> l1
    y = a ==? d;
    // CHECK: [[TMP1:%.+]] = moore.read %d : l32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.i32 -> !moore.l32
    // CHECK: moore.wildcard_eq [[TMP1]], [[TMP3]] : l32 -> l1
    y = d ==? b;
    // CHECK: [[TMP1:%.+]] = moore.read %d : l32
    // CHECK: [[TMP2:%.+]] = moore.read %e : l32
    // CHECK: moore.wildcard_eq [[TMP1]], [[TMP2]] : l32 -> l1
    y = d ==? e;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.wildcard_ne [[TMP1]], [[TMP2]] : i32 -> i1
    x = a !=? b;

    // CHECK: [[TMP1:%.+]] = moore.read %u : i32
    // CHECK: [[TMP2:%.+]] = moore.read %w : i32
    // CHECK: moore.uge [[TMP1]], [[TMP2]] : i32 -> i1
    c = u >= w;
    // CHECK: [[TMP1:%.+]] = moore.read %u : i32
    // CHECK: [[TMP2:%.+]] = moore.read %w : i32
    // CHECK: moore.ugt [[TMP1]], [[TMP2]] : i32 -> i1
    c = u > w;
    // CHECK: [[TMP1:%.+]] = moore.read %u : i32
    // CHECK: [[TMP2:%.+]] = moore.read %w : i32
    // CHECK: moore.ule [[TMP1]], [[TMP2]] : i32 -> i1
    c = u <= w;
    // CHECK: [[TMP1:%.+]] = moore.read %u : i32
    // CHECK: [[TMP2:%.+]] = moore.read %w : i32
    // CHECK: moore.ult [[TMP1]], [[TMP2]] : i32 -> i1
    c = u < w;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.sge [[TMP1]], [[TMP2]] : i32 -> i1
    c = a >= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.sgt [[TMP1]], [[TMP2]] : i32 -> i1
    c = a > b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.sle [[TMP1]], [[TMP2]] : i32 -> i1
    c = a <= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.slt [[TMP1]], [[TMP2]] : i32 -> i1
    c = a < b;

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: [[A:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast [[TMP2]] : i32 -> i1
    // CHECK: moore.and [[A]], [[B]] : i1
    c = a && b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: [[A:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast [[TMP2]] : i32 -> i1
    // CHECK: moore.or [[A]], [[B]] : i1
    c = a || b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: [[A:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast [[TMP2]] : i32 -> i1
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : i1
    // CHECK: moore.or [[NOT_A]], [[B]] : i1
    c = a -> b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: [[A:%.+]] = moore.bool_cast [[TMP1]] : i32 -> i1
    // CHECK: [[B:%.+]] = moore.bool_cast [[TMP2]] : i32 -> i1
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : i1
    // CHECK: [[NOT_B:%.+]] = moore.not [[B]] : i1
    // CHECK: [[BOTH:%.+]] = moore.and [[A]], [[B]] : i1
    // CHECK: [[NOT_BOTH:%.+]] = moore.and [[NOT_A]], [[NOT_B]] : i1
    // CHECK: moore.or [[BOTH]], [[NOT_BOTH]] : i1
    c = a <-> b;

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.shl [[TMP1]], [[TMP2]] : i32, i32
    c = a << b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.shr [[TMP1]], [[TMP2]] : i32, i32
    c = a >> b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.shl [[TMP1]], [[TMP2]] : i32, i32
    c = a <<< b;
    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.ashr [[TMP1]], [[TMP2]] : i32, i32
    c = a >>> b;
    // CHECK: [[TMP1:%.+]] = moore.read %u : i32
    // CHECK: [[TMP2:%.+]] = moore.read %b : i32
    // CHECK: moore.shr [[TMP1]], [[TMP2]] : i32, i32
    c = u >>> b;

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %a : i32
    // CHECK: moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    c = a inside { a };

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %a : i32
    // CHECK: [[TMP3:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    // CHECK: [[TMP4:%.+]] = moore.read %b : i32
    // CHECK: [[TMP5:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP4]] : i32 -> i1
    // CHECK: moore.or [[TMP3]], [[TMP5]] : i1
    c = a inside { a, b };

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %a : i32
    // CHECK: [[TMP3:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    // CHECK: [[TMP4:%.+]] = moore.read %b : i32
    // CHECK: [[TMP5:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP4]] : i32 -> i1
    // CHECK: [[TMP6:%.+]] = moore.read %a : i32
    // CHECK: [[TMP7:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP6]] : i32 -> i1
    // CHECK: [[TMP8:%.+]] = moore.read %b : i32
    // CHECK: [[TMP9:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP8]] : i32 -> i1
    // CHECK: [[TMP10:%.+]] = moore.or [[TMP7]], [[TMP9]] : i1
    // CHECK: [[TMP11:%.+]] = moore.or [[TMP5]], [[TMP10]] : i1
    // CHECK: moore.or [[TMP3]], [[TMP11]] : i1
    c = a inside { a, b, a, b };

    // CHECK: [[TMP1:%.+]] = moore.read %a : i32
    // CHECK: [[TMP2:%.+]] = moore.read %a : i32
    // CHECK: [[TMP3:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP2]] : i32 -> i1
    // CHECK: [[TMP4:%.+]] = moore.read %b : i32
    // CHECK: [[TMP5:%.+]] = moore.wildcard_eq [[TMP1]], [[TMP4]] : i32 -> i1
    // CHECK: [[TMP6:%.+]] = moore.read %a : i32
    // CHECK: [[TMP7:%.+]] = moore.read %b : i32
    // CHECK: [[TMP8:%.+]] = moore.sge [[TMP1]], [[TMP6]] : i32 -> i1
    // CHECK: [[TMP9:%.+]] = moore.sle [[TMP1]], [[TMP7]] : i32 -> i1
    // CHECK: [[TMP10:%.+]] = moore.and [[TMP8]], [[TMP9]] : i1
    // CHECK: [[TMP11:%.+]] = moore.or [[TMP5]], [[TMP10]] : i1
    // CHECK: moore.or [[TMP3]], [[TMP11]] : i1
    c = a inside { a, b, [a:b] };

    //===------------------------------------------------------------------===//
    // Conditional operator

    // CHECK: [[X_COND:%.+]] = moore.read %x : i1
    // CHECK: moore.conditional [[X_COND]] : i1 -> i32 {
    // CHECK:   [[A_READ:%.+]] = moore.read %a : i32
    // CHECK:   moore.yield [[A_READ]] : i32
    // CHECK: } {
    // CHECK:   [[B_READ:%.+]] = moore.read %b : i32
    // CHECK:   moore.yield [[B_READ]] : i32
    // CHECK: }
    c = x ? a : b;

    // CHECK: [[X_COND:%.+]] = moore.read %x : i1
    // CHECK: moore.conditional [[X_COND]] : i1 -> real {
    // CHECK:   [[R1_READ:%.+]] = moore.read %r1 : real
    // CHECK:   moore.yield [[R1_READ]] : real
    // CHECK: } {
    // CHECK:   [[R2_READ:%.+]] = moore.read %r2 : real
    // CHECK:   moore.yield [[R2_READ]] : real
    // CHECK: }
    r1 = x ? r1 : r2;

    // CHECK: [[A_COND:%.+]] = moore.read %a : i32
    // CHECK: [[TMP1:%.+]] = moore.bool_cast [[A_COND]] : i32 -> i1
    // CHECK: moore.conditional [[TMP1]] : i1 -> i32 {
    // CHECK:   [[A_READ:%.+]] = moore.read %a : i32
    // CHECK:   moore.yield [[A_READ]] : i32
    // CHECK: } {
    // CHECK:   [[B_READ:%.+]] = moore.read %b : i32
    // CHECK:   moore.yield [[B_READ]] : i32
    // CHECK: }
    c = a ? a : b;

    // CHECK: [[A_SGT:%.+]] = moore.read %a : i32
    // CHECK: [[B_SGT:%.+]] = moore.read %b : i32
    // CHECK: [[TMP1:%.+]] = moore.sgt [[A_SGT]], [[B_SGT]] : i32 -> i1
    // CHECK: moore.conditional [[TMP1]] : i1 -> i32 {
    // CHECK:   [[A_ADD:%.+]] = moore.read %a : i32
    // CHECK:   [[B_ADD:%.+]] = moore.read %b : i32
    // CHECK:   [[TMP2:%.+]] = moore.add [[A_ADD]], [[B_ADD]] : i32
    // CHECK:   moore.yield [[TMP2]] : i32
    // CHECK: } {
    // CHECK:   [[A_SUB:%.+]] = moore.read %a : i32
    // CHECK:   [[B_SUB:%.+]] = moore.read %b : i32
    // CHECK:   [[TMP2:%.+]] = moore.sub [[A_SUB]], [[B_SUB]] : i32
    // CHECK:   moore.yield [[TMP2]] : i32
    // CHECK: }
    c = (a > b) ? (a + b) : (a - b);

    //===------------------------------------------------------------------===//
    // Assign operators

    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.add [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a += b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.sub [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a -= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.mul [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a *= b;
    // CHECK: [[TMP1:%.+]] = moore.read %f
    // CHECK: [[TMP2:%.+]] = moore.read %d
    // CHECK: [[TMP3:%.+]] = moore.divs [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %f, [[TMP3]]
    f /= d;
    // CHECK: [[TMP1:%.+]] = moore.read %g
    // CHECK: [[TMP2:%.+]] = moore.read %h
    // CHECK: [[TMP3:%.+]] = moore.divu [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %g, [[TMP3]]
    g /= h;
    // CHECK: [[TMP1:%.+]] = moore.read %f
    // CHECK: [[TMP2:%.+]] = moore.read %d
    // CHECK: [[TMP3:%.+]] = moore.mods [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %f, [[TMP3]]
    f %= d;
    // CHECK: [[TMP1:%.+]] = moore.read %g
    // CHECK: [[TMP2:%.+]] = moore.read %h
    // CHECK: [[TMP3:%.+]] = moore.modu [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %g, [[TMP3]]
    g %= h;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.and [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a &= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.or [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a |= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.xor [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a ^= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.shl [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a <<= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.shl [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a <<<= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.shr [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a >>= b;
    // CHECK: [[TMP1:%.+]] = moore.read %a
    // CHECK: [[TMP2:%.+]] = moore.read %b
    // CHECK: [[TMP3:%.+]] = moore.ashr [[TMP1]], [[TMP2]]
    // CHECK: moore.blocking_assign %a, [[TMP3]]
    a >>>= b;

    // CHECK: [[A_ADD:%.+]] = moore.read %a
    // CHECK: [[A_MUL:%.+]] = moore.read %a
    // CHECK: [[A_DEC:%.+]] = moore.read %a
    // CHECK: [[TMP1:%.+]] = moore.constant 1
    // CHECK: [[TMP2:%.+]] = moore.sub [[A_DEC]], [[TMP1]]
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    // CHECK: [[TMP1:%.+]] = moore.mul [[A_MUL]], [[A_DEC]]
    // CHECK: moore.blocking_assign %a, [[TMP1]]
    // CHECK: [[TMP2:%.+]] = moore.add [[A_ADD]], [[TMP1]]
    // CHECK: moore.blocking_assign %a, [[TMP2]]
    a += (a *= a--);
 
  end
endmodule

// CHECK-LABEL: moore.module @Conversion
module Conversion;
  // Implicit conversion.
  // CHECK: %a = moore.variable
  // CHECK: [[TMP1:%.+]] = moore.read %a : i16
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i16 -> !moore.i32
  // CHECK: %b = moore.variable [[TMP2]]
  shortint a;
  int b = a;

  // Explicit conversion.
  // CHECK: [[TMP1:%.+]] = moore.read %a : i16
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i16 -> !moore.i8
  // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.i8 -> !moore.i32
  // CHECK: %c = moore.variable [[TMP3]]
  int c = byte'(a);

  // Sign conversion.
  // CHECK: [[TMP1:%.+]] = moore.read %b : i32
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.i32
  // CHECK: %d1 = moore.variable [[TMP2]]
  // CHECK: [[TMP3:%.+]] = moore.read %b : i32
  // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.i32 -> !moore.i32
  // CHECK: %d2 = moore.variable [[TMP4]]
  bit signed [31:0] d1 = signed'(b);
  bit [31:0] d2 = unsigned'(b);

  // Width conversion.
  // CHECK: [[TMP1:%.+]] = moore.read %b : i32
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.i32 -> !moore.i19
  // CHECK: %e = moore.variable [[TMP2]]
  bit signed [18:0] e = 19'(b);
endmodule

// CHECK-LABEL: moore.module @PortsTop
module PortsTop;
  wire x0, y0, z0;
  logic w0;
  // CHECK: [[x0:%.+]] = moore.read %x0 : l1
  // CHECK: [[B:%.+]] = moore.instance "p0" @PortsAnsi(
  // CHECK-SAME:   a: [[x0]]: !moore.l1
  // CHECK-SAME:   c: %z0: !moore.ref<l1>
  // CHECK-SAME:   d: %w0: !moore.ref<l1>
  // CHECK-SAME: ) -> (b: !moore.l1)
  // CHECK-NEXT: moore.assign %y0, [[B]]
  PortsAnsi p0(x0, y0, z0, w0);

  wire x1, y1, z1;
  logic w1;
  // CHECK: [[x1:%.+]] = moore.read %x1 : l1
  // CHECK: [[B:%.+]] = moore.instance "p1" @PortsNonAnsi(
  // CHECK-SAME:   a: [[x1]]: !moore.l1
  // CHECK-SAME:   c: %z1: !moore.ref<l1>
  // CHECK-SAME:   d: %w1: !moore.ref<l1>
  // CHECK-SAME: ) -> (b: !moore.l1)
  // CHECK-NEXT: moore.assign %y1, [[B]]
  PortsNonAnsi p1(x1, y1, z1, w1);

  wire x2;
  wire [1:0] y2;
  int z2;
  wire w2, v2;
  // CHECK: [[X2:%.+]] = moore.read %x2
  // CHECK: [[Y2:%.+]] = moore.read %y2
  // CHECK: [[B0:%.+]], [[B1:%.+]], [[B2:%.+]] = moore.instance "p2" @PortsExplicit(
  // CHECK-SAME:   a0: [[X2]]: !moore.l1
  // CHECK-SAME:   a1: [[Y2]]: !moore.l2
  // CHECK-SAME: ) -> (
  // CHECK-SAME:   b0: !moore.i32
  // CHECK-SAME:   b1: !moore.l1
  // CHECK-SAME:   b2: !moore.l1
  // CHECK-SAME: )
  // CHECK-NEXT: moore.assign %z2, [[B0]]
  // CHECK-NEXT: moore.assign %w2, [[B1]]
  // CHECK-NEXT: moore.assign %v2, [[B2]]
  PortsExplicit p2(x2, y2, z2, w2, v2);

  wire x3, y3;
  wire [2:0] z3;
  wire [1:0] w3;
  // CHECK: [[X3:%.+]] = moore.read %x3
  // CHECK: [[Y3:%.+]] = moore.read %y3
  // CHECK: [[TMP:%.+]] = moore.constant 0 :
  // CHECK: [[V2:%.+]] = moore.extract_ref %z3 from [[TMP]]
  // CHECK: [[TMP:%.+]] = moore.constant 1 :
  // CHECK: [[V1:%.+]] = moore.extract_ref %z3 from [[TMP]]
  // CHECK: [[TMP:%.+]] = moore.constant 2 :
  // CHECK: [[V0:%.+]] = moore.extract_ref %z3 from [[TMP]]
  // CHECK: [[V0_READ:%.+]] = moore.read [[V0]]
  // CHECK: [[TMP:%.+]] = moore.constant 0 :
  // CHECK: [[C1:%.+]] = moore.extract_ref %w3 from [[TMP]]
  // CHECK: [[TMP:%.+]] = moore.constant 1 :
  // CHECK: [[C0:%.+]] = moore.extract_ref %w3 from [[TMP]]
  // CHECK: [[C0_READ:%.+]] = moore.read [[C0]]
  // CHECK: [[V1_VALUE:%.+]], [[C1_VALUE:%.+]] = moore.instance "p3" @MultiPorts(
  // CHECK-SAME:   a0: [[X3]]: !moore.l1
  // CHECK-SAME:   a1: [[Y3]]: !moore.l1
  // CHECK-SAME:   v0: [[V0_READ]]: !moore.l1
  // CHECK-SAME:   v2: [[V2]]: !moore.ref<l1>
  // CHECK-SAME:   c0: [[C0_READ]]: !moore.l1
  // CHECK-SAME: ) -> (
  // CHECK-SAME:   v1: !moore.l1
  // CHECK-SAME:   c1: !moore.l1
  // CHECK-SAME: )
  // CHECK-NEXT: moore.assign [[V1]], [[V1_VALUE]]
  // CHECK-NEXT: moore.assign [[C1]], [[C1_VALUE]]
  MultiPorts p3(x3, y3, z3, w3);

  wire x4, y4;
  // CHECK: %a = moore.net wire : <l1>
  // CHECK: [[A_VALUE:%.+]] = moore.read %a : l1
  // CHECK: [[X4:%.+]] = moore.read %x4 : l1
  // CHECK: %c = moore.variable : <l1>
  // CHECK: [[C_VALUE:%.+]] = moore.read %c : l1
  // CHECK: [[D_VALUE:%.+]], [[E_VALUE:%.+]] = moore.instance "p4" @PortsUnconnected(
  // CHECK-SAME: a: [[A_VALUE]]: !moore.l1
  // CHECK-SAME: b: [[X4]]: !moore.l1
  // CHECK-SAME: c: [[C_VALUE]]: !moore.l1
  // CHECK-SAME: ) -> (
  // CHECK-SAME: d: !moore.l1
  // CHECK-SAME: e: !moore.l1
  // CHECK-SAME: )
  // CHECK: moore.assign %y4, [[D_VALUE]] : l1
  PortsUnconnected p4(.a(), .b(x4), .c(), .d(y4), .e());
endmodule

// CHECK-LABEL: moore.module private @PortsAnsi
module PortsAnsi(
  // CHECK-SAME: in %a : !moore.l1
  input a,
  // CHECK-SAME: out b : !moore.l1
  output b,
  // CHECK-SAME: in %c : !moore.ref<l1>
  inout c,
  // CHECK-SAME: in %d : !moore.ref<l1>
  ref d
);
  // Internal nets and variables created by Slang for each port.
  // CHECK: [[A_INT:%.+]] = moore.net name "a" wire : <l1>
  // CHECK: [[B_INT:%.+]] = moore.net wire : <l1>
  // CHECK: [[C_INT:%.+]] = moore.net name "c" wire : <l1>
  // CHECK: [[D_INT:%.+]] = moore.variable name "d" : <l1>

  // Mapping ports to local declarations.
  // CHECK: moore.assign [[A_INT]], %a : l1
  // CHECK: [[B_READ:%.+]] = moore.read %b : l1
  // CHECK: [[C_READ:%.+]] = moore.read %c : l1
  // CHECK: moore.assign [[C_INT]], [[C_READ]] : l1
  // CHECK: [[D_READ:%.+]] = moore.read %d : l1
  // CHECK: moore.assign [[D_INT]], [[D_READ]] : l1
  // CHECK: moore.output [[B_READ]] : !moore.l1
endmodule

// CHECK-LABEL: moore.module private @PortsNonAnsi
module PortsNonAnsi(a, b, c, d);
  // CHECK-SAME: in %a : !moore.l1
  input a;
  // CHECK-SAME: out b : !moore.l1
  output b;
  // CHECK-SAME: in %c : !moore.ref<l1>
  inout c;
  // CHECK-SAME: in %d : !moore.ref<l1>
  ref logic d;
endmodule

// CHECK-LABEL: moore.module private @PortsExplicit
module PortsExplicit(
  // CHECK-SAME: in %a0 : !moore.l1
  input .a0(x),
  // CHECK-SAME: in %a1 : !moore.l2
  input .a1({y, z}),
  // CHECK-SAME: out b0 : !moore.i32
  output .b0(42),
  // CHECK-SAME: out b1 : !moore.l1
  output .b1(x),
  // CHECK-SAME: out b2 : !moore.l1
  output .b2(y ^ z)
);
  logic x, y, z;

  // Input mappings
  // CHECK: moore.assign %x, %a0
  // CHECK: [[TMP:%.+]] = moore.concat_ref %y, %z
  // CHECK: moore.assign [[TMP]], %a1

  // Output mappings
  // CHECK: [[B0:%.+]] = moore.constant 42
  // CHECK: [[X_READ:%.+]] = moore.read %x : l1
  // CHECK: [[Y_READ:%.+]] = moore.read %y : l1
  // CHECK: [[Z_READ:%.+]] = moore.read %z : l1
  // CHECK: [[B2:%.+]] = moore.xor [[Y_READ]], [[Z_READ]]
  // CHECK: moore.output [[B0]], [[X_READ]], [[B2]]
endmodule

// CHECK-LABEL: moore.module private @MultiPorts
module MultiPorts(
  // CHECK-SAME: in %a0 : !moore.l1
  .a0(u[0]),
  // CHECK-SAME: in %a1 : !moore.l1
  .a1(u[1]),
  // CHECK-SAME: in %v0 : !moore.l1
  // CHECK-SAME: out v1 : !moore.l1
  // CHECK-SAME: in %v2 : !moore.ref<l1>
  .b({v0, v1, v2}),
  // CHECK-SAME: in %c0 : !moore.l1
  // CHECK-SAME: out c1 : !moore.l1
  {c0, c1}
);
  // CHECK: [[V0:%.+]] = moore.net name "v0" wire
  // CHECK: [[V1:%.+]] = moore.net wire
  // CHECK: [[V2:%.+]] = moore.net name "v2" wire
  // CHECK: [[C0:%.+]] = moore.net name "c0" wire
  // CHECK: [[C1:%.+]] = moore.net wire
  input [1:0] u;
  input v0;
  output v1;
  inout v2;
  input c0;
  output c1;

  // CHECK: [[TMP1:%.+]] = moore.constant 0 :
  // CHECK: [[TMP2:%.+]] = moore.extract_ref %u from [[TMP1]]
  // CHECK: moore.assign [[TMP2]], %a0

  // CHECK: [[TMP1:%.+]] = moore.constant 1 :
  // CHECK: [[TMP2:%.+]] = moore.extract_ref %u from [[TMP1]]
  // CHECK: moore.assign [[TMP2]], %a1

  // CHECK: moore.assign [[V0]], %v0
  // CHECK: [[V1_READ:%.+]] = moore.read %v1
  // CHECK: [[V2_READ:%.+]] = moore.read %v2
  // CHECK: moore.assign [[V2]], [[V2_READ]]
  // CHECK: moore.assign [[C0]], %c0
  // CHECK: [[C1_READ:%.+]] = moore.read %c1
  // CHECK: moore.output [[V1_READ]], [[C1_READ]]
endmodule

// CHECK-LABEL: moore.module private @PortsUnconnected
module PortsUnconnected(
  // CHECK-SAME: in %a : !moore.l1
  input a,
  // CHECK-SAME: in %b : !moore.l1
  input b,
  // CHECK-SAME: in %c : !moore.l1
  input logic c,
  // CHECK-SAME: out d : !moore.l1
  output d,
  // CHECK-SAME: out e : !moore.l1
  output e
);
  // Internal nets and variables created by Slang for each port.
  // CHECK: [[A_INT:%.+]] = moore.net name "a" wire : <l1>
  // CHECK: [[B_INT:%.+]] = moore.net name "b" wire : <l1>
  // CHECK: [[C_INT:%.+]] = moore.variable name "c" : <l1>
  // CHECK: [[D_INT:%.+]] = moore.net wire : <l1>
  // CHECK: [[E_INT:%.+]] = moore.net wire : <l1>
  
  // Mapping ports to local declarations.
  // CHECK: moore.assign [[A_INT]], %a : l1
  // CHECK: moore.assign [[B_INT]], %b : l1
  // CHECK: [[D_READ:%.+]] = moore.read [[D_INT]] : l1
  // CHECK: [[E_READ:%.+]] = moore.read [[E_INT]] : l1
  // CHECK: moore.output [[D_READ]], [[E_READ]] : !moore.l1, !moore.l1
endmodule

// CHECK-LABEL: moore.module @EventControl(in %clk : !moore.l1)
module EventControl(input clk);
  // CHECK: %clk_0 = moore.net name "clk" wire : <l1>

  int a1, a2, b, c;

  // CHECK: moore.procedure always
  // CHECK:   [[CLK_READ:%.+]] = moore.read %clk_0 : l1
  // CHECK:   moore.wait_event posedge [[CLK_READ]] : l1
  always @(posedge clk) begin end;

  // CHECK: moore.procedure always
  // CHECK:   [[CLK_READ:%.+]] = moore.read %clk_0 : l1
  // CHECK:   moore.wait_event negedge [[CLK_READ]] : l1
  always @(negedge clk) begin end;

  // CHECK: moore.procedure always
  // CHECK:   [[CLK_READ:%.+]] = moore.read %clk_0 : l1
  // CHECK:   moore.wait_event edge [[CLK_READ]] : l1
  always @(edge clk) begin end;

  // CHECK: moore.procedure always {
  // CHECK:   [[B_READ:%.+]] = moore.read %b : i32
  // CHECK:   moore.wait_event none [[B_READ]] : i32
  // CHECK:   [[C_READ:%.+]] = moore.read %c : i32
  // CHECK:   moore.wait_event none [[C_READ]] : i32
  always @(b, c) begin
    // CHECK: [[B_READ:%.+]] = moore.read %b : i32
    // CHECK: [[C_READ:%.+]] = moore.read %c : i32
    // CHECK: [[ADD:%.+]] = moore.add [[B_READ]], [[C_READ]] : i32
    // CHECK: moore.blocking_assign %a1, [[ADD]] : i32
    a1 = b + c;
  end;

  // CHECK: moore.procedure always
  always @(*) begin
    // CHECK: [[B_READ:%.+]] = moore.read %b : i32
    // CHECK: [[C_READ:%.+]] = moore.read %c : i32
    // CHECK: [[ADD:%.+]] = moore.add [[B_READ]], [[C_READ]] : i32
    // CHECK: moore.blocking_assign %a2, [[ADD]] : i32
    a2 = b + c;
  end

  // CHECK: moore.assign %clk_0, %clk : l1
  // CHECK: moore.output
endmodule

// CHECK-LABEL: moore.module @GenerateConstructs()
module GenerateConstructs;
  genvar i;
  parameter p=2;
  
  generate
    // CHECK: [[TMP1:%.+]] = moore.constant 0 : l32
    // CHECK: %i = moore.named_constant localparam [[TMP1]] : l32
    // CHECK: [[TMP2:%.+]] = moore.conversion %i : !moore.l32 -> !moore.i32
    // CHECK: %g1 = moore.variable [[TMP2]] : <i32>
    // CHECK: [[TMP3:%.+]] = moore.constant 1 : l32
    // CHECK: %i_0 = moore.named_constant localparam name "i" [[TMP3]] : l32
    // CHECK: [[TMP4:%.+]] = moore.conversion %i_0 : !moore.l32 -> !moore.i32
    // CHECK: %g1_1 = moore.variable name "g1" [[TMP4]] : <i32>
    for(i=0; i<2; i=i+1) begin
      int g1 = i;
    end

    // CHECK: [[TMP:%.+]] = moore.constant 2 : i32
    // CHECK: %g2 = moore.variable [[TMP]] : <i32>
    if(p == 2) begin
      int g2 = 2;
    end
    else begin
      int g2 = 3;
    end
    
    // CHECK: [[TMP:%.+]] = moore.constant 2 : i32
    // CHECK: %g3 = moore.variable [[TMP]] : <i32>
    case (p)
      2: begin
        int g3 = 2;
        end
      default: begin
        int g3 = 3;
        end
    endcase
  endgenerate
endmodule
