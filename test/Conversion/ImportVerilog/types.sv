// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// CHECK-LABEL: moore.module @Enums
module Enums;
  typedef enum shortint { MAGIC } myEnum;

  // CHECK-NEXT: %e0 = moore.variable : !moore.i32
  // CHECK-NEXT: %e1 = moore.variable : !moore.i8
  // CHECK-NEXT: %e2 = moore.variable : !moore.i16
  enum { FOO, BAR } e0;
  enum byte { HELLO = 0, WORLD = 1 } e1;
  myEnum e2;
endmodule

// CHECK-LABEL: moore.module @IntAtoms
module IntAtoms;
  // CHECK-NEXT: %d0 = moore.variable : !moore.l1
  // CHECK-NEXT: %d1 = moore.variable : !moore.i1
  // CHECK-NEXT: %d2 = moore.variable : !moore.l1
  // CHECK-NEXT: %d3 = moore.variable : !moore.i32
  // CHECK-NEXT: %d4 = moore.variable : !moore.i16
  // CHECK-NEXT: %d5 = moore.variable : !moore.i64
  // CHECK-NEXT: %d6 = moore.variable : !moore.l32
  // CHECK-NEXT: %d7 = moore.variable : !moore.i8
  // CHECK-NEXT: %d8 = moore.variable : !moore.l64
  logic d0;
  bit d1;
  reg d2;
  int d3;
  shortint d4;
  longint d5;
  integer d6;
  byte d7;
  time d8;

  // CHECK-NEXT: %u0 = moore.variable : !moore.l1
  // CHECK-NEXT: %u1 = moore.variable : !moore.i1
  // CHECK-NEXT: %u2 = moore.variable : !moore.l1
  // CHECK-NEXT: %u3 = moore.variable : !moore.i32
  // CHECK-NEXT: %u4 = moore.variable : !moore.i16
  // CHECK-NEXT: %u5 = moore.variable : !moore.i64
  // CHECK-NEXT: %u6 = moore.variable : !moore.l32
  // CHECK-NEXT: %u7 = moore.variable : !moore.i8
  // CHECK-NEXT: %u8 = moore.variable : !moore.l64
  logic unsigned u0;
  bit unsigned u1;
  reg unsigned u2;
  int unsigned u3;
  shortint unsigned u4;
  longint unsigned u5;
  integer unsigned u6;
  byte unsigned u7;
  time unsigned u8;

  // CHECK-NEXT: %s0 = moore.variable : !moore.l1
  // CHECK-NEXT: %s1 = moore.variable : !moore.i1
  // CHECK-NEXT: %s2 = moore.variable : !moore.l1
  // CHECK-NEXT: %s3 = moore.variable : !moore.i32
  // CHECK-NEXT: %s4 = moore.variable : !moore.i16
  // CHECK-NEXT: %s5 = moore.variable : !moore.i64
  // CHECK-NEXT: %s6 = moore.variable : !moore.l32
  // CHECK-NEXT: %s7 = moore.variable : !moore.i8
  // CHECK-NEXT: %s8 = moore.variable : !moore.l64
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

// CHECK-LABEL: moore.module @MultiPackedRangeDim
module MultiPackedRangeDim;
  // CHECK-NEXT: %v0 = moore.variable : !moore.packed<range<l3, 5:0>>
  // CHECK-NEXT: %v1 = moore.variable : !moore.packed<range<l3, 0:5>>
  logic [5:0][2:0] v0;
  logic [0:5][2:0] v1;
endmodule

// CHECK-LABEL: moore.module @MultiUnpackedRangeDim
module MultiUnpackedRangeDim;
  // CHECK-NEXT: %v0 = moore.variable : !moore.unpacked<range<range<l1, 2:0>, 5:0>>
  // CHECK-NEXT: %v1 = moore.variable : !moore.unpacked<range<range<l1, 2:0>, 0:5>>
  logic v0 [5:0][2:0];
  logic v1 [0:5][2:0];
endmodule

// CHECK-LABEL: moore.module @MultiUnpackedUnsizedDim
module MultiUnpackedUnsizedDim;
  // CHECK-NEXT: %v0 = moore.variable : !moore.unpacked<unsized<unsized<l1>>>
  logic v0 [][];
endmodule

// CHECK-LABEL: moore.module @PackedRangeDim
module PackedRangeDim;
  // CHECK-NEXT: %d0 = moore.variable : !moore.l3
  // CHECK-NEXT: %d1 = moore.variable : !moore.l3
  logic [2:0] d0;
  logic [0:2] d1;
endmodule

// CHECK-LABEL: moore.module @RealType
module RealType;
  // CHECK-NEXT: %d0 = moore.variable : !moore.real
  // CHECK-NEXT: %d1 = moore.variable : !moore.realtime
  // CHECK-NEXT: %d2 = moore.variable : !moore.shortreal
  real d0;
  realtime d1;
  shortreal d2;
endmodule

// CHECK-LABEL: moore.module @Structs
module Structs;
  typedef struct packed { byte a; int b; } myStructA;
  typedef struct { byte x; int y; } myStructB;

  // CHECK-NEXT: %s0 = moore.variable : !moore.packed<struct<{foo: i1, bar: l1}>>
  // CHECK-NEXT: %s1 = moore.variable : !moore.unpacked<struct<{many: assoc<i1, i32>}>>
  // CHECK-NEXT: %s2 = moore.variable : !moore.packed<struct<{a: i8, b: i32}>>
  // CHECK-NEXT: %s3 = moore.variable : !moore.unpacked<struct<{x: i8, y: i32}>>
  struct packed { bit foo; logic bar; } s0;
  struct { bit many[int]; } s1;
  myStructA s2;
  myStructB s3;
endmodule

// CHECK-LABEL: moore.module @Typedefs
module Typedefs;
  typedef logic [2:0] myType1;
  typedef logic myType2 [2:0];

  // CHECK-NEXT: %v0 = moore.variable : !moore.l3
  // CHECK-NEXT: %v1 = moore.variable : !moore.unpacked<range<l1, 2:0>>
  myType1 v0;
  myType2 v1;
endmodule

// CHECK-LABEL: moore.module @UnpackedAssocDim
module UnpackedAssocDim;
  // CHECK-NEXT: %d0 = moore.variable : !moore.unpacked<assoc<l1, i32>>
  // CHECK-NEXT: %d1 = moore.variable : !moore.unpacked<assoc<l1, l1>>
  logic d0 [int];
  logic d1 [logic];
endmodule

// CHECK-LABEL: moore.module @UnpackedQueueDim
module UnpackedQueueDim;
  //CHECK-NEXT: %d0 = moore.variable : !moore.unpacked<queue<l1, 0>>
  //CHECK-NEXT: %d1 = moore.variable : !moore.unpacked<queue<l1, 2>>
  logic d0[$];
  logic d1[$:2];
endmodule

// CHECK-LABEL: moore.module @UnpackedRangeDim
module UnpackedRangeDim;
  // CHECK-NEXT: %d0 = moore.variable : !moore.unpacked<range<l1, 2:0>>
  // CHECK-NEXT: %d1 = moore.variable : !moore.unpacked<range<l1, 0:2>>
  logic d0 [2:0];
  logic d1 [0:2];
endmodule

// CHECK-LABEL: moore.module @UnpackedUnsizedDim
module UnpackedUnsizedDim;
  // CHECK-NEXT: %d0 = moore.variable : !moore.unpacked<unsized<l1>>
  logic d0 [];
endmodule
