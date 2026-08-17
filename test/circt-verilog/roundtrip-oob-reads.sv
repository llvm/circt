// RUN: circt-verilog -Wno-error=range-oob %s | circt-opt --test-apply-lowering-options='options=emittedLineLength=0' --export-verilog -o /dev/null | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: module vector(
module vector(input bit [1:0] i, output bit [4:0] o);
  // CHECK: assign o = {2'h0, i, 1'h0};
  assign o = i[3:-1];
endmodule

// CHECK-LABEL: module packed_array(
module packed_array(input bit [1:0][1:0] i, output bit [4:0][1:0] o);
  // CHECK: [[MSB_PAD:[A-Za-z0-9_]+]] = '{2'h0, 2'h0};
  // CHECK-NEXT: [[LSB_PAD:[A-Za-z0-9_]+]] = '{2'h0};
  // CHECK-NEXT: assign o = {[[MSB_PAD]], i, [[LSB_PAD]]};
  assign o = i[3:-1];
endmodule

// CHECK-LABEL: module packed_struct_array(
typedef struct packed { bit hi; bit lo; } pair_t;
module packed_struct_array(input pair_t [1:0] i, output pair_t [4:0] o);
  // CHECK: [[MSB_PAD:[A-Za-z0-9_]+]] = '{'{hi: 1'h0, lo: 1'h0}, '{hi: 1'h0, lo: 1'h0}};
  // CHECK-NEXT: [[LSB_PAD:[A-Za-z0-9_]+]] = '{'{hi: 1'h0, lo: 1'h0}};
  // CHECK-NEXT: assign o = {[[MSB_PAD]], i, [[LSB_PAD]]};
  assign o = i[3:-1];
endmodule
