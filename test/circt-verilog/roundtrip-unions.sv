// RUN: circt-verilog %s | circt-opt -export-verilog -o /dev/null | FileCheck %s

typedef union packed {
  int x;
  int y;
} my_union_t;

// CHECK-LABEL: module top
// CHECK: union packed {logic [31:0] x;logic [31:0] y;}
module top(input my_union_t a, output int b);
  // CHECK: assign b = a.y;
  assign b = a.y;
endmodule

typedef struct packed {
  logic [15:0] high;
  logic [15:0] low;
} my_struct_t;

typedef union packed {
  int x;
  my_struct_t s;
} my_union_2_t;

// CHECK-LABEL: module top2
// CHECK: union packed {logic [31:0] x;struct packed {logic [15:0] high; logic [15:0] low; } s;}
module top2(input my_union_2_t a, output logic [15:0] b);
  // CHECK: assign b = a.s.high;
  assign b = a.s.high;
endmodule

typedef union packed {
  int x;
  logic [3:0][7:0] bytes;
} my_union_3_t;

// CHECK-LABEL: module top3
// CHECK: union packed {logic [31:0] x;logic [3:0][7:0] bytes;}
module top3(input my_union_3_t a, output logic [7:0] b);
  // CHECK: assign b = a.bytes[2'h2];
  assign b = a.bytes[2];
endmodule
