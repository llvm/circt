// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: hw.module @ClockPosEdgeAlwaysFF(
module ClockPosEdgeAlwaysFF(input logic clock, input int d, output int q);
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg %d, [[CLK]] : i32
  // CHECK: hw.output [[REG]]
  always_ff @(posedge clock) q <= d;
endmodule

// CHECK-LABEL: hw.module @ClockPosEdge(
module ClockPosEdge(input logic clock, input int d, output int q);
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg %d, [[CLK]] : i32
  // CHECK: hw.output [[REG]]
  always @(posedge clock) q <= d;
endmodule

// CHECK-LABEL: hw.module @ClockNegEdge(
module ClockNegEdge(input logic clock, input int d, output int q);
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[CLK_INV:%.+]] = seq.clock_inv [[CLK]]
  // CHECK: [[REG:%.+]] = seq.compreg %d, [[CLK_INV]] : i32
  // CHECK: hw.output [[REG]]
  always @(negedge clock) q <= d;
endmodule

// CHECK-LABEL: hw.module @ActiveHighReset(
module ActiveHighReset(input logic clock, input logic reset, input int d1, input int d2, output int q1, output int q2);
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG1:%.+]] = seq.compreg %d1, [[CLK]] reset %reset, %c42_i32 : i32
  // CHECK: [[REG2:%.+]] = seq.compreg %d2, [[CLK]] reset %reset, %c42_i32 : i32
  // CHECK: hw.output [[REG1]], [[REG2]]
  always @(posedge clock, posedge reset) if (reset) q1 <= 42; else q1 <= d1;
  always @(posedge clock, posedge reset) q2 <= reset ? 42 : d2;
endmodule

// CHECK-LABEL: hw.module @ActiveLowReset(
module ActiveLowReset(input logic clock, input logic reset, input int d1, input int d2, output int q1, output int q2);
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[RST_INV:%.+]] = comb.xor %reset, %true
  // CHECK: [[REG1:%.+]] = seq.compreg %d1, [[CLK]] reset [[RST_INV]], %c42_i32 : i32
  // CHECK: [[REG2:%.+]] = seq.compreg %d2, [[CLK]] reset [[RST_INV]], %c42_i32 : i32
  // CHECK: hw.output [[REG1]], [[REG2]]
  always @(posedge clock, negedge reset) if (!reset) q1 <= 42; else q1 <= d1;
  always @(posedge clock, negedge reset) q2 <= !reset ? 42 : d2;
endmodule

// CHECK-LABEL: hw.module @Enable(
module Enable(input logic clock, input logic enable, input int d, output int q);
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg.ce %d, [[CLK]], %enable : i32
  // CHECK: hw.output [[REG]]
  always @(posedge clock) if (enable) q <= d;
endmodule

// CHECK-LABEL: hw.module @ResetAndEnable(
module ResetAndEnable(input logic clock, input logic reset, input logic enable, input int d, output int q);
  // CHECK: [[CLK:%.+]] = seq.to_clock %clock
  // CHECK: [[REG:%.+]] = seq.compreg.ce %d, [[CLK]], %enable reset %reset, %c42_i32 : i32
  // CHECK: hw.output [[REG]]
  always @(posedge clock, posedge reset) if (reset) q <= 42; else if (enable) q <= d;
endmodule
