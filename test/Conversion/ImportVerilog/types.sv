// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @Enums
module Enums;
  typedef enum shortint { MAGIC } myEnum;

  // CHECK-NEXT: %e0 = moore.variable : <i32>
  // CHECK-NEXT: %e1 = moore.variable : <i8>
  // CHECK-NEXT: %e2 = moore.variable : <i16>
  enum { FOO, BAR } e0;
  enum byte { HELLO = 0, WORLD = 1 } e1;
  myEnum e2;
endmodule

// CHECK-LABEL: moore.module @IntAtoms
module IntAtoms;
  // CHECK-NEXT: %d0 = moore.variable : <l1>
  // CHECK-NEXT: %d1 = moore.variable : <i1>
  // CHECK-NEXT: %d2 = moore.variable : <l1>
  // CHECK-NEXT: %d3 = moore.variable : <i32>
  // CHECK-NEXT: %d4 = moore.variable : <i16>
  // CHECK-NEXT: %d5 = moore.variable : <i64>
  // CHECK-NEXT: %d6 = moore.variable : <l32>
  // CHECK-NEXT: %d7 = moore.variable : <i8>
  // CHECK-NEXT: %d8 = moore.variable : <time>
  logic d0;
  bit d1;
  reg d2;
  int d3;
  shortint d4;
  longint d5;
  integer d6;
  byte d7;
  time d8;

  // CHECK-NEXT: %u0 = moore.variable : <l1>
  // CHECK-NEXT: %u1 = moore.variable : <i1>
  // CHECK-NEXT: %u2 = moore.variable : <l1>
  // CHECK-NEXT: %u3 = moore.variable : <i32>
  // CHECK-NEXT: %u4 = moore.variable : <i16>
  // CHECK-NEXT: %u5 = moore.variable : <i64>
  // CHECK-NEXT: %u6 = moore.variable : <l32>
  // CHECK-NEXT: %u7 = moore.variable : <i8>
  // CHECK-NEXT: %u8 = moore.variable : <time>
  logic unsigned u0;
  bit unsigned u1;
  reg unsigned u2;
  int unsigned u3;
  shortint unsigned u4;
  longint unsigned u5;
  integer unsigned u6;
  byte unsigned u7;
  time unsigned u8;

  // CHECK-NEXT: %s0 = moore.variable : <l1>
  // CHECK-NEXT: %s1 = moore.variable : <i1>
  // CHECK-NEXT: %s2 = moore.variable : <l1>
  // CHECK-NEXT: %s3 = moore.variable : <i32>
  // CHECK-NEXT: %s4 = moore.variable : <i16>
  // CHECK-NEXT: %s5 = moore.variable : <i64>
  // CHECK-NEXT: %s6 = moore.variable : <l32>
  // CHECK-NEXT: %s7 = moore.variable : <i8>
  // CHECK-NEXT: %s8 = moore.variable : <time>
  logic signed s0;
  bit signed s1;
  reg signed s2;
  int signed s3;
  shortint signed s4;
  longint signed s5;
  integer signed s6;
  byte signed s7;
  time signed s8;
endmodule

// CHECK-LABEL: moore.module @Dimensions
module Dimensions;
  // CHECK-NEXT: %p0 = moore.variable : <l3>
  logic [2:0] p0;
  // CHECK-NEXT: %p1 = moore.variable : <l3>
  logic [0:2] p1;
  // CHECK-NEXT: %p2 = moore.variable : <array<6 x l3>>
  logic [5:0][2:0] p2;
  // CHECK-NEXT: %p3 = moore.variable : <array<6 x l3>>
  logic [0:5][2:0] p3;

  // CHECK-NEXT: %u0 = moore.variable : <uarray<3 x l1>>
  logic u0 [2:0];
  // CHECK-NEXT: %u1 = moore.variable : <uarray<3 x l1>>
  logic u1 [0:2];
  // CHECK-NEXT: %u2 = moore.variable : <uarray<6 x uarray<3 x l1>>>
  logic u2 [5:0][2:0];
  // CHECK-NEXT: %u3 = moore.variable : <uarray<6 x uarray<3 x l1>>>
  logic u3 [0:5][2:0];
  // CHECK-NEXT: %u4 = moore.variable : <open_uarray<l1>>
  logic u4 [];
  // CHECK-NEXT: %u5 = moore.variable : <open_uarray<open_uarray<l1>>>
  logic u5 [][];
  // CHECK-NEXT: %u6 = moore.variable : <uarray<42 x l1>>
  logic u6 [42];
  // CHECK-NEXT: %u7 = moore.variable : <assoc_array<l1, i32>>
  logic u7 [int];
  // CHECK-NEXT: %u8 = moore.variable : <assoc_array<l1, l1>>
  logic u8 [logic];
  //CHECK-NEXT: %u9 = moore.variable : <queue<l1, 0>>
  logic u9 [$];
  //CHECK-NEXT: %u10 = moore.variable : <queue<l1, 2>>
  logic u10 [$:2];
endmodule

// CHECK-LABEL: moore.module @RealType
module RealType;
  // CHECK-NEXT: %d0 = moore.variable : <real>
  real d0;
  // CHECK-NEXT: %d1 = moore.variable : <time>
  realtime d1;
  // CHECK-NEXT: %d2 = moore.variable : <real>
  shortreal d2;
endmodule

// CHECK-LABEL: moore.module @Structs
module Structs;
  typedef struct packed { byte a; int b; } myStructA;
  typedef struct { byte x; int y; } myStructB;

  // CHECK-NEXT: %s0 = moore.variable : <struct<{foo: i1, bar: l1}>>
  // CHECK-NEXT: %s1 = moore.variable : <ustruct<{many: assoc_array<i1, i32>}>>
  // CHECK-NEXT: %s2 = moore.variable : <struct<{a: i8, b: i32}>>
  // CHECK-NEXT: %s3 = moore.variable : <ustruct<{x: i8, y: i32}>>
  struct packed { bit foo; logic bar; } s0;
  struct { bit many[int]; } s1;
  myStructA s2;
  myStructB s3;
endmodule

// CHECK-LABEL: moore.module @Typedefs
module Typedefs;
  typedef logic [2:0] myType1;
  typedef logic myType2 [2:0];

  // CHECK-NEXT: %v0 = moore.variable : <l3>
  // CHECK-NEXT: %v1 = moore.variable : <uarray<3 x l1>>
  myType1 v0;
  myType2 v1;
endmodule

// CHECK-LABEL: moore.module @String
module String;
  // CHECK-NEXT: %s = moore.variable : <string>
  string s;
endmodule
