// RUN: circt-verilog --import-only %s | FileCheck %s
// RUN: circt-verilog %s -o /dev/null
// REQUIRES: slang
// UNSUPPORTED: valgrind

module bits_child(output logic bits[2]);
  assign bits[0] = 1'b0;
  assign bits[1] = 1'b1;
endmodule

// The first declared array element drives values[0], not values[1].
// CHECK-LABEL: moore.module @positional()
// CHECK: %[[VALUES:.*]] = moore.variable : <l2>
// CHECK: %[[BITS:.*]] = moore.instance "c" @bits_child
// CHECK: %[[LOW:.*]] = moore.extract_ref %[[VALUES]] from 0
// CHECK: %[[HIGH:.*]] = moore.extract_ref %[[VALUES]] from 1
// CHECK: %[[FIRST:.*]] = moore.extract %[[BITS]] from 1
// CHECK-NEXT: moore.assign %[[LOW]], %[[FIRST]] : l1
// CHECK: %[[SECOND:.*]] = moore.extract %[[BITS]] from 0
// CHECK-NEXT: moore.assign %[[HIGH]], %[[SECOND]] : l1
module positional;
  logic [1:0] values;
  bits_child c(.bits('{values[0], values[1]}));
endmodule

// CHECK-LABEL: moore.module @direct()
// CHECK: %[[VALUES:.*]] = moore.variable : <uarray<2 x l1>>
// CHECK: %[[BITS:.*]] = moore.instance "c" @bits_child
// CHECK-NEXT: moore.assign %[[VALUES]], %[[BITS]] : uarray<2 x l1>
module direct;
  logic values[2];
  bits_child c(.bits(values));
endmodule

module array_child(output logic signed [3:0] a[5:3]);
  assign a = '{4'sh8, 4'sh3, 4'sh5};
endmodule

// Non-zero descending bounds, signed extension, a part select, and truncation.
// CHECK-LABEL: moore.module @wide()
// CHECK: %[[X:.*]] = moore.variable : <l8>
// CHECK: %[[Y:.*]] = moore.variable : <l8>
// CHECK: %[[Z:.*]] = moore.variable : <l2>
// CHECK: %[[A:.*]] = moore.instance "c" @array_child
// CHECK: %[[PART:.*]] = moore.extract_ref %[[Y]] from 0
// CHECK-NEXT: %[[FIRST:.*]] = moore.extract %[[A]] from 2
// CHECK-NEXT: %[[EXT:.*]] = moore.sext %[[FIRST]] : l4 -> l8
// CHECK-NEXT: moore.assign %[[X]], %[[EXT]] : l8
// CHECK-NEXT: %[[SECOND:.*]] = moore.extract %[[A]] from 1
// CHECK-NEXT: moore.assign %[[PART]], %[[SECOND]] : l4
// CHECK-NEXT: %[[THIRD:.*]] = moore.extract %[[A]] from 0
// CHECK-NEXT: %[[TRUNC:.*]] = moore.trunc %[[THIRD]] : l4 -> l2
// CHECK-NEXT: moore.assign %[[Z]], %[[TRUNC]] : l2
module wide;
  logic signed [7:0] x;
  logic [7:0] y;
  logic [1:0] z;
  array_child c(.a('{x, y[3:0], z}));
endmodule

module nested_child(output logic [3:0] a[-2:-1][7:6]);
  assign a = '{'{4'h1, 4'h2}, '{4'h3, 4'h4}};
endmodule

// Nested patterns with negative ascending and positive descending bounds.
// CHECK-LABEL: moore.module @nested()
// CHECK: %[[X:.*]] = moore.variable : <l16>
// CHECK: %[[A:.*]] = moore.instance "c" @nested_child
// CHECK: %[[D0:.*]] = moore.extract_ref %[[X]] from 0
// CHECK: %[[D1:.*]] = moore.extract_ref %[[X]] from 4
// CHECK: %[[D2:.*]] = moore.extract_ref %[[X]] from 8
// CHECK: %[[D3:.*]] = moore.extract_ref %[[X]] from 12
// CHECK-NEXT: %[[ROW0:.*]] = moore.extract %[[A]] from 1
// CHECK-NEXT: %[[V0:.*]] = moore.extract %[[ROW0]] from 1
// CHECK-NEXT: moore.assign %[[D0]], %[[V0]] : l4
// CHECK-NEXT: %[[V1:.*]] = moore.extract %[[ROW0]] from 0
// CHECK-NEXT: moore.assign %[[D1]], %[[V1]] : l4
// CHECK-NEXT: %[[ROW1:.*]] = moore.extract %[[A]] from 0
// CHECK-NEXT: %[[V2:.*]] = moore.extract %[[ROW1]] from 1
// CHECK-NEXT: moore.assign %[[D2]], %[[V2]] : l4
// CHECK-NEXT: %[[V3:.*]] = moore.extract %[[ROW1]] from 0
// CHECK-NEXT: moore.assign %[[D3]], %[[V3]] : l4
module nested;
  logic [15:0] x;
  nested_child c(.a('{'{x[3:0], x[7:4]}, '{x[11:8], x[15:12]}}));
endmodule

typedef struct { logic [2:0] a; logic signed [3:0] b; } pair_t;
module struct_child(output pair_t p);
  assign p = '{3'd3, -4'sd2};
endmodule

// CHECK-LABEL: moore.module @structs()
// CHECK: %[[A:.*]] = moore.variable : <l3>
// CHECK: %[[B:.*]] = moore.variable : <l8>
// CHECK: %[[P:.*]] = moore.instance "c" @struct_child
// CHECK-NEXT: %[[F0:.*]] = moore.struct_extract %[[P]], "a"
// CHECK-NEXT: moore.assign %[[A]], %[[F0]] : l3
// CHECK-NEXT: %[[F1:.*]] = moore.struct_extract %[[P]], "b"
// CHECK-NEXT: %[[EXT:.*]] = moore.sext %[[F1]] : l4 -> l8
// CHECK-NEXT: moore.assign %[[B]], %[[EXT]] : l8
module structs;
  logic [2:0] a;
  logic signed [7:0] b;
  struct_child c(.p('{a, b}));
endmodule

module input_child(input logic [3:0] a[2]);
endmodule

// Input pattern elements still produce values, including keyed patterns.
// CHECK-LABEL: moore.module @inputs()
// CHECK: %[[X:.*]] = moore.variable : <l4>
// CHECK: %[[Y:.*]] = moore.variable : <l4>
// CHECK: %[[XV:.*]] = moore.read %[[X]] : <l4>
// CHECK: %[[YV:.*]] = moore.read %[[Y]] : <l4>
// CHECK: %[[ARRAY:.*]] = moore.array_create %[[XV]], %[[YV]]
// CHECK: moore.instance "c" @input_child(a: %[[ARRAY]]:
// CHECK: %[[XV2:.*]] = moore.read %[[X]] : <l4>
// CHECK: %[[YV2:.*]] = moore.read %[[Y]] : <l4>
// CHECK: %[[ARRAY2:.*]] = moore.array_create %[[XV2]], %[[YV2]]
// CHECK: moore.instance "d" @input_child(a: %[[ARRAY2]]:
module inputs;
  logic [3:0] x, y;
  input_child c(.a('{x, y}));
  input_child d(.a('{1: y, 0: x}));
endmodule

module packed_child(output logic [4:3][3:0] a);
  assign a = 8'h12;
endmodule

// Packed array outputs can drive selected unpacked array elements.
// CHECK-LABEL: moore.module @packed_array()
// CHECK: %[[D:.*]] = moore.variable : <uarray<2 x l4>>
// CHECK: %[[A:.*]] = moore.instance "c" @packed_child
// CHECK: %[[D0:.*]] = moore.extract_ref %[[D]] from 1
// CHECK: %[[D1:.*]] = moore.extract_ref %[[D]] from 0
// CHECK-NEXT: %[[V0:.*]] = moore.extract %[[A]] from 1 : array<2 x l4> -> l4
// CHECK-NEXT: moore.assign %[[D0]], %[[V0]] : l4
// CHECK-NEXT: %[[V1:.*]] = moore.extract %[[A]] from 0 : array<2 x l4> -> l4
// CHECK-NEXT: moore.assign %[[D1]], %[[V1]] : l4
module packed_array;
  logic [3:0] d[9:10];
  packed_child c(.a('{d[9], d[10]}));
endmodule

typedef struct packed { logic a; logic [2:0] b; } packed_pair_t;
module packed_struct_child(output packed_pair_t p);
  assign p = 4'h9;
endmodule

// CHECK-LABEL: moore.module @packed_struct()
// CHECK: %[[A:.*]] = moore.variable : <l1>
// CHECK: %[[B:.*]] = moore.variable : <l3>
// CHECK: %[[P:.*]] = moore.instance "c" @packed_struct_child
// CHECK-NEXT: %[[F0:.*]] = moore.struct_extract %[[P]], "a"
// CHECK-NEXT: moore.assign %[[A]], %[[F0]] : l1
// CHECK-NEXT: %[[F1:.*]] = moore.struct_extract %[[P]], "b"
// CHECK-NEXT: moore.assign %[[B]], %[[F1]] : l3
module packed_struct;
  logic a;
  logic [2:0] b;
  packed_struct_child c(.p('{a, b}));
endmodule

module vector_child(output logic [3:2] a);
  assign a = 2'b01;
endmodule

// CHECK-LABEL: moore.module @vector_pattern()
// CHECK: %[[A:.*]] = moore.variable : <l1>
// CHECK: %[[B:.*]] = moore.variable : <l1>
// CHECK: %[[V:.*]] = moore.instance "c" @vector_child
// CHECK-NEXT: %[[V0:.*]] = moore.extract %[[V]] from 1 : l2 -> l1
// CHECK-NEXT: moore.assign %[[A]], %[[V0]] : l1
// CHECK-NEXT: %[[V1:.*]] = moore.extract %[[V]] from 0 : l2 -> l1
// CHECK-NEXT: moore.assign %[[B]], %[[V1]] : l1
module vector_pattern;
  logic a, b;
  vector_child c(.a('{a, b}));
endmodule

module inout_child(inout wire a[2]);
endmodule

// Inout connections remain direct references, without output-only assignments.
// CHECK-LABEL: moore.module @direct_inout()
// CHECK: %[[A:.*]] = moore.net wire : <uarray<2 x l1>>
// CHECK-NEXT: moore.instance "c" @inout_child(a: %[[A]]: !moore.ref<uarray<2 x l1>>)
// CHECK-NEXT: moore.output
module direct_inout;
  wire a[2];
  inout_child c(.a(a));
endmodule
