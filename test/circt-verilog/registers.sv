// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: hw.module @ClockOnly
module ClockOnly(input logic clock, input int d, output int q);
  int q1, q2;
  assign q = q1 ^ q2;
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: seq.compreg %d, [[CLK]] : i32
  always @(posedge clock) q1 <= d;
  // CHECK: seq.compreg %d, [[CLK]] : i32
  always_ff @(posedge clock) q2 <= d;
endmodule

// CHECK-LABEL: hw.module @AsyncReset
module AsyncReset(input logic clock, input logic reset, input int d, output int q);
  int q1, q2;
  assign q = q1 ^ q2;
  // BROKEN: [[CLK:%.+]] = seq.to_clock %clock
  // BROKEN: seq.compreg %d, [[CLK]] reset %reset, %c42_i32 : i32
  always_ff @(posedge clock or posedge reset) if (reset) q1 <= 42; else q1 <= d;
  // BROKEN: [[RESET_INV:%.+]] = comb.xor %reset, %true
  // BROKEN: seq.compreg %d, [[CLK]] reset [[RESET_INV]], %c42_i32 : i32
  always_ff @(posedge clock or negedge reset) if (!reset) q2 <= 42; else q2 <= d;
endmodule
