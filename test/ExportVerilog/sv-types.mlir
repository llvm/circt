// RUN: circt-translate %s -export-verilog -verify-diagnostics | FileCheck %s --strict-whitespace

// CHECK: typedef logic foo;
sv.typedef @foo : i1

// CHECK: typedef struct packed {logic a; logic [22:0] b; } FancyStruct;
sv.typedef @FancyStruct : !rtl.struct<a: i1, b: i23>

// CHECK: typedef logic [7:0][0:15] unpacked_array;
sv.typedef @unpacked_array : !rtl.uarray<16xi8>
