// RUN: circt-verilog %s -E -DBAR=1337 -DHELLO -I%S/include | FileCheck %s
// REQUIRES: slang

// CHECK-NOT: define FOO
// CHECK: localparam X = 42
`define FOO 42
localparam X = `FOO;

// CHECK: localparam Y = 1337
localparam Y = `BAR;

// CHECK: localparam FOUND_HELLO;
`ifdef HELLO
localparam FOUND_HELLO;
`endif

// CHECK: localparam Z = 9001
`include "other.sv"
