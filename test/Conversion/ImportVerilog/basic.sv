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
  // CHECK: %v0 = moore.variable : !moore.logic
  // CHECK: %v1 = moore.variable : !moore.int
  // CHECK: %v2 = moore.variable %v1 : !moore.int
  var v0;
  int v1;
  int v2 = v1;

  // CHECK: %n0 = moore.net "wire" : !moore.logic
  // CHECK: %n1 = moore.net "tri" : !moore.logic
  // CHECK: %n2 = moore.net "wire" : !moore.int
  wire n0;
  tri n1;
  wire integer n2;

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

  // CHECK: moore.assign %v1, %v2 : !moore.int
  assign v1 = v2;
endmodule

// CHECK-LABEL: moore.module @Statements
module Statements;
  bit x, y, z;
  int i;
  initial begin
    //===------------------------------------------------------------------===//
    // Conditional statements

    // CHECK: [[COND:%.+]] = moore.conversion %x : !moore.bit -> i1
    // CHECK: scf.if [[COND]] {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK: }
    if (x) x = y;

    // CHECK: [[COND0:%.+]] = moore.and %x, %y
    // CHECK: [[COND1:%.+]] = moore.conversion [[COND0]] : !moore.bit -> i1
    // CHECK: scf.if [[COND1]] {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK: }
    if (x &&& y) x = y;

    // CHECK: [[COND:%.+]] = moore.conversion %x : !moore.bit -> i1
    // CHECK: scf.if [[COND]] {
    // CHECK:   moore.blocking_assign %x, %z
    // CHECK: } else {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK: }
    if (x) x = z; else x = y;

    // CHECK: [[COND:%.+]] = moore.conversion %x : !moore.bit -> i1
    // CHECK: scf.if [[COND]] {
    // CHECK:   moore.blocking_assign %x, %x
    // CHECK: } else {
    // CHECK:   [[COND:%.+]] = moore.conversion %y : !moore.bit -> i1
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
    // Loop statements

    // CHECK: moore.blocking_assign %y, %x
    // CHECK: scf.while : () -> () {
    // CHECK:   [[COND:%.+]] = moore.conversion %x : !moore.bit -> i1
    // CHECK:   scf.condition([[COND]])
    // CHECK: } do {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK:   moore.blocking_assign %x, %z
    // CHECK:   scf.yield
    // CHECK: }
    for (y = x; x; x = z) x = y;

    // CHECK: scf.while (%arg0 = %i) : (!moore.int) -> !moore.int {
    // CHECK:   [[TMP0:%.+]] = moore.bool_cast %arg0 : !moore.int -> !moore.bit
    // CHECK:   [[TMP1:%.+]] = moore.conversion [[TMP0]] : !moore.bit -> i1
    // CHECK:   scf.condition([[TMP1]]) %arg0 : !moore.int
    // CHECK: } do {
    // CHECK: ^bb0(%arg0: !moore.int):
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK:   [[TMP0:%.+]] = moore.constant 1 : !moore.int
    // CHECK:   [[TMP1:%.+]] = moore.sub %arg0, [[TMP0]] : !moore.int
    // CHECK:   scf.yield [[TMP1]] : !moore.int
    // CHECK: }
    repeat (i) x = y;

    // CHECK: scf.while : () -> () {
    // CHECK:   [[COND:%.+]] = moore.conversion %x : !moore.bit -> i1
    // CHECK:   scf.condition([[COND]])
    // CHECK: } do {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK:   scf.yield
    // CHECK: }
    while (x) x = y;

    // CHECK: scf.while : () -> () {
    // CHECK:   moore.blocking_assign %x, %y
    // CHECK:   [[COND:%.+]] = moore.conversion %x : !moore.bit -> i1
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

    // CHECK: moore.blocking_assign %x, %y : !moore.bit
    x = y;

    // CHECK: moore.blocking_assign %y, %z : !moore.bit
    // CHECK: moore.blocking_assign %x, %z : !moore.bit
    x = (y = z);

    // CHECK: moore.nonblocking_assign %x, %y : !moore.bit
    x <= y;
  end
endmodule

// CHECK-LABEL: moore.module @Expressions {
module Expressions;
  // CHECK: %a = moore.variable : !moore.int
  // CHECK: %b = moore.variable : !moore.int
  // CHECK: %c = moore.variable : !moore.int
  int a, b, c;
  int unsigned u;
  bit [1:0][3:0] v;
  integer d, e, f;
  bit x;
  logic y;

  initial begin
    // CHECK: moore.constant 0 : !moore.packed<range<bit, 31:0>>
    c = '0;
    // CHECK: moore.constant -1 : !moore.packed<range<bit, 31:0>>
    c = '1;
    // CHECK: moore.constant 42 : !moore.int
    c = 42;
    // CHECK: moore.constant 42 : !moore.packed<range<bit, 18:0>>
    c = 19'd42;
    // CHECK: moore.constant 42 : !moore.packed<range<bit<signed>, 18:0>>
    c = 19'sd42;
    // CHECK: moore.concat %a, %b, %c : (!moore.int, !moore.int, !moore.int) -> !moore.packed<range<bit, 95:0>>
    a = {a, b, c};
    // CHECK: moore.concat %d, %e : (!moore.integer, !moore.integer) -> !moore.packed<range<logic, 63:0>>
    d = {d, e};

    //===------------------------------------------------------------------===//
    // Unary operators

    // CHECK: moore.blocking_assign %c, %a : !moore.int
    c = +a;
    // CHECK: moore.neg %a : !moore.int
    c = -a;
    // CHECK: [[TMP1:%.+]] = moore.conversion %v : !moore.packed<range<range<bit, 3:0>, 1:0>> -> !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP2:%.+]] = moore.neg [[TMP1]] : !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP3:%.+]] = moore.conversion [[TMP2]] : !moore.packed<range<bit, 31:0>> -> !moore.int
    c = -v;
    // CHECK: moore.not %a : !moore.int
    c = ~a;
    // CHECK: moore.reduce_and %a : !moore.int -> !moore.bit
    x = &a;
    // CHECK: moore.reduce_and %d : !moore.integer -> !moore.logic
    y = &d;
    // CHECK: moore.reduce_or %a : !moore.int -> !moore.bit
    x = |a;
    // CHECK: moore.reduce_xor %a : !moore.int -> !moore.bit
    x = ^a;
    // CHECK: [[TMP:%.+]] = moore.reduce_and %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ~&a;
    // CHECK: [[TMP:%.+]] = moore.reduce_or %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ~|a;
    // CHECK: [[TMP:%.+]] = moore.reduce_xor %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ~^a;
    // CHECK: [[TMP:%.+]] = moore.reduce_xor %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = ^~a;
    // CHECK: [[TMP:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: moore.not [[TMP]] : !moore.bit
    x = !a;
    // CHECK: [[PRE:%.+]] = moore.read_lvalue %a : !moore.int
    // CHECK: [[TMP:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[POST:%.+]] = moore.add [[PRE]], [[TMP]] : !moore.int
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[PRE]]
    c = a++;
    // CHECK: [[PRE:%.+]] = moore.read_lvalue %a : !moore.int
    // CHECK: [[TMP:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[POST:%.+]] = moore.sub [[PRE]], [[TMP]] : !moore.int
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[PRE]]
    c = a--;
    // CHECK: [[PRE:%.+]] = moore.read_lvalue %a : !moore.int
    // CHECK: [[TMP:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[POST:%.+]] = moore.add [[PRE]], [[TMP]] : !moore.int
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[POST]]
    c = ++a;
    // CHECK: [[PRE:%.+]] = moore.read_lvalue %a : !moore.int
    // CHECK: [[TMP:%.+]] = moore.constant 1 : !moore.int
    // CHECK: [[POST:%.+]] = moore.sub [[PRE]], [[TMP]] : !moore.int
    // CHECK: moore.blocking_assign %a, [[POST]]
    // CHECK: moore.blocking_assign %c, [[POST]]
    c = --a;

    //===------------------------------------------------------------------===//
    // Binary operators

    // CHECK: moore.add %a, %b : !moore.int
    c = a + b;
    // CHECK: [[TMP1:%.+]] = moore.conversion %a : !moore.int -> !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP2:%.+]] = moore.conversion %v : !moore.packed<range<range<bit, 3:0>, 1:0>> -> !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP3:%.+]] = moore.add [[TMP1]], [[TMP2]] : !moore.packed<range<bit, 31:0>>
    // CHECK: [[TMP4:%.+]] = moore.conversion [[TMP3]] : !moore.packed<range<bit, 31:0>> -> !moore.int
    c = a + v;
    // CHECK: moore.sub %a, %b : !moore.int
    c = a - b;
    // CHECK: moore.mul %a, %b : !moore.int
    c = a * b;
    // CHECK: moore.div %d, %e : !moore.integer
    f = d / e;
    // CHECK: moore.mod %d, %e : !moore.integer
    f = d % e;

    // CHECK: moore.and %a, %b : !moore.int
    c = a & b;
    // CHECK: moore.or %a, %b : !moore.int
    c = a | b;
    // CHECK: moore.xor %a, %b : !moore.int
    c = a ^ b;
    // CHECK: [[TMP:%.+]] = moore.xor %a, %b : !moore.int
    // CHECK: moore.not [[TMP]] : !moore.int
    c = a ~^ b;
    // CHECK: [[TMP:%.+]] = moore.xor %a, %b : !moore.int
    // CHECK: moore.not [[TMP]] : !moore.int
    c = a ^~ b;

    // CHECK: moore.eq %a, %b : !moore.int -> !moore.bit
    x = a == b;
    // CHECK: moore.eq %d, %e : !moore.integer -> !moore.logic
    y = d == e;
    // CHECK: moore.ne %a, %b : !moore.int -> !moore.bit
    x = a != b ;
    // CHECK: moore.case_eq %a, %b : !moore.int
    x = a === b;
    // CHECK: moore.case_ne %a, %b : !moore.int
    x = a !== b;
    // CHECK: moore.wildcard_eq %a, %b : !moore.int -> !moore.bit
    x = a ==? b;
    // CHECK: [[TMP:%.+]] = moore.conversion %a : !moore.int -> !moore.integer
    // CHECK: moore.wildcard_eq [[TMP]], %d : !moore.integer -> !moore.logic
    y = a ==? d;
    // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.integer
    // CHECK: moore.wildcard_eq %d, [[TMP]] : !moore.integer -> !moore.logic
    y = d ==? b;
    // CHECK: moore.wildcard_eq %d, %e : !moore.integer -> !moore.logic
    y = d ==? e;
    // CHECK: moore.wildcard_ne %a, %b : !moore.int -> !moore.bit
    x = a !=? b;

    // CHECK: moore.ge %a, %b : !moore.int -> !moore.bit
    c = a >= b;
    // CHECK: moore.gt %a, %b : !moore.int -> !moore.bit
    c = a > b;
    // CHECK: moore.le %a, %b : !moore.int -> !moore.bit
    c = a <= b;
    // CHECK: moore.lt %a, %b : !moore.int -> !moore.bit
    c = a < b;

    // CHECK: [[A:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[B:%.+]] = moore.bool_cast %b : !moore.int -> !moore.bit
    // CHECK: moore.and [[A]], [[B]] : !moore.bit
    c = a && b;
    // CHECK: [[A:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[B:%.+]] = moore.bool_cast %b : !moore.int -> !moore.bit
    // CHECK: moore.or [[A]], [[B]] : !moore.bit
    c = a || b;
    // CHECK: [[A:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[B:%.+]] = moore.bool_cast %b : !moore.int -> !moore.bit
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : !moore.bit
    // CHECK: moore.or [[NOT_A]], [[B]] : !moore.bit
    c = a -> b;
    // CHECK: [[A:%.+]] = moore.bool_cast %a : !moore.int -> !moore.bit
    // CHECK: [[B:%.+]] = moore.bool_cast %b : !moore.int -> !moore.bit
    // CHECK: [[NOT_A:%.+]] = moore.not [[A]] : !moore.bit
    // CHECK: [[NOT_B:%.+]] = moore.not [[B]] : !moore.bit
    // CHECK: [[BOTH:%.+]] = moore.and [[A]], [[B]] : !moore.bit
    // CHECK: [[NOT_BOTH:%.+]] = moore.and [[NOT_A]], [[NOT_B]] : !moore.bit
    // CHECK: moore.or [[BOTH]], [[NOT_BOTH]] : !moore.bit
    c = a <-> b;

    // CHECK: moore.shl %a, %b : !moore.int, !moore.int
    c = a << b;
    // CHECK: moore.shr %a, %b : !moore.int, !moore.int
    c = a >> b;
    // CHECK: moore.shl %a, %b : !moore.int, !moore.int
    c = a <<< b;
    // CHECK: moore.ashr %a, %b : !moore.int, !moore.int
    c = a >>> b;
    // CHECK: moore.shr %u, %b : !moore.int<unsigned>, !moore.int
    c = u >>> b;

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
    // CHECK: [[TMP2:%.+]] = moore.div [[TMP1]], %d
    // CHECK: moore.blocking_assign %f, [[TMP2]]
    f /= d;
    // CHECK: [[TMP1:%.+]] = moore.read_lvalue %f
    // CHECK: [[TMP2:%.+]] = moore.mod [[TMP1]], %d
    // CHECK: moore.blocking_assign %f, [[TMP2]]
    f %= d;
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
  end
endmodule

// CHECK-LABEL: moore.module @Conversion {
module Conversion;
  // Implicit conversion.
  // CHECK: %a = moore.variable
  // CHECK: [[TMP:%.+]] = moore.conversion %a : !moore.shortint -> !moore.int
  // CHECK: %b = moore.variable [[TMP]]
  shortint a;
  int b = a;

  // Explicit conversion.
  // CHECK: [[TMP1:%.+]] = moore.conversion %a : !moore.shortint -> !moore.byte
  // CHECK: [[TMP2:%.+]] = moore.conversion [[TMP1]] : !moore.byte -> !moore.int
  // CHECK: %c = moore.variable [[TMP2]]
  int c = byte'(a);

  // Sign conversion.
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.packed<range<bit<signed>, 31:0>>
  // CHECK: %d1 = moore.variable [[TMP]]
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.packed<range<bit, 31:0>>
  // CHECK: %d2 = moore.variable [[TMP]]
  bit signed [31:0] d1 = signed'(b);
  bit [31:0] d2 = unsigned'(b);

  // Width conversion.
  // CHECK: [[TMP:%.+]] = moore.conversion %b : !moore.int -> !moore.packed<range<bit<signed>, 18:0>>
  // CHECK: %e = moore.variable [[TMP]]
  bit signed [18:0] e = 19'(b);
endmodule
