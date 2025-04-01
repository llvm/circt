; RUN: firtool %s -disable-all-randomization -split-input-file | FileCheck %s

; This is an end-to-end example of a test-bench (Foo) enabling verification,
; probing into a device-under-test (Bar), and reading from hardware which is
; only present if the verification layer is enabled.

FIRRTL version 4.0.0

circuit Foo: %[[
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Foo|Bar>c"},
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Foo|Foo>d"},
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Foo|VerificationHelper>w"},
  {"class": "circt.OutputDirAnnotation", "target": "~Foo|Foo", "dirname": "testbench"}
]]
  ; Collateral for the Verification layer should be output into the testbench directory.
  layer Verification, bind, "testbench":

  ; Since VerificationHelper is only used under the Verification layer, it
  ; should be output in the testbench directory too.
  module VerificationHelper:
    wire w : UInt<1>
    invalidate w

  ; Although the Component Bar is only instantiated by testbench Foo, since Bar
  ; is public, it remains in the default build directory.
  public module Bar:
    input a: UInt<1>
    output b: Probe<UInt<1>, Verification>

    ; The extracted Bar_Verification module should be placed into the
    ; testbench output directory.
    layerblock Verification:
      node c = UInt<1>(0)
      define b = probe(c)
      inst helper of VerificationHelper

  ; The TestBench module is explicitly annotated to be placed in the testbench
  ; output directory.
  public module Foo enablelayer Verification:
    inst bar of Bar
    node d = read(bar.b)
    connect bar.a, d

  ; CHECK: module Bar(
  ; CHECK:  input a
  ; CHECK: );
  ; CHECK: endmodule

  ; CHECK: FILE "testbench{{[/\]}}layers-Foo-Verification.sv"
  ; CHECK: `ifndef layers_Foo_Verification
  ; CHECK: `define layers_Foo_Verification
  ; CHECK: bind Bar Bar_Verification verification ();
  ; CHECK: `endif // layers_Foo_Verification

  ; CHECK: FILE "testbench{{[/\]}}VerificationHelper.sv"
  ; CHECK: module VerificationHelper();
  ; CHECK:   wire w = 1'h0;
  ; CHECK: endmodule

  ; CHECK: FILE "testbench{{[/\]}}Bar_Verification.sv"
  ; CHECK: module Bar_Verification();
  ; CHECK:   wire c = 1'h0;
  ; CHECK:   wire c_probe = c;
  ; CHECK: endmodule

  ; CHECK: FILE "testbench{{[/\]}}Foo.sv"
  ; CHECK: module Foo();
  ; CHECK:  wire d = Foo.bar.verification.c_probe;
  ; CHECK:   Bar bar (
  ; CHECK:   .a (d)
  ; CHECK:   );
  ; CHECK: endmodule

; // -----

; This is an end-to-end example of a test-harness enabling verification, probing
; into a device-under-test, and reading from hardware which is only present if
; the verification layer is enabled.

FIRRTL version 4.0.0

circuit TestHarness:

  layer Verification, bind:

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
  ; CHECK:         $fwrite(`PRINTF_FD_, "The last PC was: %x", dut_trace);
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

; CHECK: FILE "layers-TestHarness-Verification.sv"
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
