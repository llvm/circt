; RUN: firtool -verilog -disable-all-randomization %s | FileCheck %s

; This is an end-to-end example of a test-harness enabling verification, probing
; into a device-under-test, and reading from hardware which is only present if
; the verification layer is enabled.

FIRRTL version 4.0.0

circuit TestHarness:

  layer Verification bind:

  ; CHECK: module DUT_Verification(
  ; CHECK:   input        _clock,
  ; CHECK:   input [31:0] _a
  ; CHECK: ); 
  ; CHECK:   reg  [31:0] pc_d;
  ; CHECK:   wire [31:0] pc_d_probe = pc_d;
  ; CHECK:   always @(posedge _clock)
  ; CHECK:     pc_d <= _a;
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
  ; CHECK:   input [31:0] _dut_trace,
  ; CHECK:   input        _clock,
  ; CHECK:                _reset
  ; CHECK: );
  ; CHECK:   `ifndef SYNTHESIS
  ; CHECK:     always @(posedge _clock) begin
  ; CHECK:       if ((`PRINTF_COND_) & _reset)
  ; CHECK:         $fwrite(32'h80000002, "The last PC was: %x", _dut_trace);
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
  module TestHarness:
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

; CHECK: // ----- 8< ----- FILE "groups_TestHarness_Verification.sv" ----- 8< -----
; CHECK: `ifndef groups_TestHarness_Verification
; CHECK: `define groups_TestHarness_Verification
; CHECK: bind DUT DUT_Verification dUT_Verification (
; CHECK:   ._clock (clock),
; CHECK:   ._a     (a)
; CHECK: );
; CHECK: bind TestHarness TestHarness_Verification testHarness_Verification (
; CHECK:   ._dut_trace (TestHarness.dut.dUT_Verification.pc_d_probe),
; CHECK:   ._clock     (clock),
; CHECK:   ._reset     (reset)
; CHECK: );
; CHECK: `endif // groups_TestHarness_Verification
