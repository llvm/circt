; RUN: firtool -verilog -disable-all-randomization %s | FileCheck %s

; This is an end-to-end example of a test-harness enabling verification, probing
; into a device-under-test, and reading from hardware which is only present if
; the verification layer is enabled.

FIRRTL version 4.0.0

circuit TestHarness:

  layer Verification bind:

  ; CHECK: module DUT_Verification(
  ; CHECK:   input        clock,
  ; CHECK:   input [31:0] a
  ; CHECK: );
  ; CHECK:   reg  [31:0] pc_d;
  ; CHECK:   wire [31:0] pc_d_probe = pc_d;
  ; CHECK:   always @(posedge clock)
  ; CHECK:     pc_d <= a;
  ; CHECK: endmodule

  ; CHECK: module DUT(
  ; CHECK:   input         clock,
  ; CHECK:   input  [31:0] a,
  ; CHECK:   output [31:0] b
  ; CHECK: );
  ; CHECK:   reg [31:0] pc;
  ; CHECK:   always @(posedge clock)
  ; CHECK:     pc <= a;
  ; CHECK:   assign b = pc;
  ; CHECK: endmodule
  module DUT:
    input clock: Clock
    input reset: UInt<1>
    input a: UInt<32>
    output b: UInt<32>
    output trace: Probe<UInt<32>, Verification>

    reg pc: UInt<32>, clock
    connect pc, a
    connect b, pc

    wire x : Probe<UInt<32>, Verification>

    layerblock Verification:
      reg pc_d: UInt<32>, clock
      connect pc_d, a
      define x = probe(pc_d)

    layerblock Verification:
      define trace = x

  ; CHECK: module TestHarness_Verification(
  ; CHECK:   input [31:0] dut_trace,
  ; CHECK:   input        clock,
  ; CHECK:                reset
  ; CHECK: );
  ; CHECK:   `ifndef SYNTHESIS
  ; CHECK:     always @(posedge clock) begin
  ; CHECK:       if ((`PRINTF_COND_) & reset)
  ; CHECK:         $fwrite(32'h80000002, "The last PC was: %x", dut_trace);
  ; CHECK:     end // always @(posedge)
  ; CHECK:   `endif // not def SYNTHESIS
  ; CHECK: endmodule

  ; CHECK: module TestHarness(
  ; CHECK:   input         clock,
  ; CHECK:                 reset,
  ; CHECK:   input  [31:0] a,
  ; CHECK:   output [31:0] b
  ; CHECK: );
  ; CHECK:   DUT dut (
  ; CHECK:     .clock (clock),
  ; CHECK:     .a     (a),
  ; CHECK:     .b     (b)
  ; CHECK:   );
  ; CHECK: endmodule
  public module TestHarness:
    input clock: Clock
    input reset: UInt<1>
    input a: UInt<32>
    output b: UInt<32>

    inst dut of DUT
    connect dut.clock, clock
    connect dut.reset, reset
    connect dut.a, a
    connect b, dut.b

    layerblock Verification:
      printf(clock, reset, "The last PC was: %x", read(dut.trace))

; CHECK: // ----- 8< ----- FILE "layers_TestHarness_Verification.sv" ----- 8< -----
; CHECK: `ifndef layers_TestHarness_Verification
; CHECK: `define layers_TestHarness_Verification
; CHECK: bind DUT DUT_Verification verification (
; CHECK:   .clock (clock),
; CHECK:   .a     (a)
; CHECK: );
; CHECK: bind TestHarness TestHarness_Verification verification (
; CHECK:   .dut_trace (TestHarness.dut.verification.pc_d_probe),
; CHECK:   .clock     (clock),
; CHECK:   .reset     (reset)
; CHECK: );
; CHECK: `endif // layers_TestHarness_Verification
