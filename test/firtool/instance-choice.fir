; RUN: firtool %s | FileCheck %s

FIRRTL version 5.1.0
circuit top:
  option Platform:
    FPGA

  module DefaultTarget:
    output result: UInt<32>[1]

    connect result[0], UInt<32>(0)

  module FPGATarget:
    output result: UInt<32>[1]

    connect result[0], UInt<32>(20)

  public module top:
    input clk: Clock
    input rst: UInt<1>
    ; CHECK:       wire [31:0] [[RESULT:.+]];
    ; CHECK-NEXT:  `ifdef [[OPTION_NAME:.+]]
    ; CHECK-NEXT:    FPGATarget proc_FPGA (
    ; CHECK-NEXT:      .result_0 ([[RESULT]])
    ; CHECK-NEXT:    );
    ; CHECK-NEXT:  `else
    ; CHECK-NEXT:    DefaultTarget proc_default (
    ; CHECK-NEXT:      .result_0 ([[RESULT]])
    ; CHECK-NEXT:    );
    ; CHECK-NEXT:    `ifndef [[TARGET_NAME:.+]]
    ; CHECK-NEXT:      `define [[TARGET_NAME]] proc_default
    ; CHECK-NEXT:    `endif
    ; CHECK-NEXT:  `endif
    instchoice proc of DefaultTarget, Platform:
      FPGA => FPGATarget

    printf(clk, UInt<1>(1), "result: %d\n", proc.result[0])
; CHECK-LABEL: // ----- 8< ----- FILE "targets-top-Platform-FPGA.svh" ----- 8< -----
; CHECK:       `ifndef [[OPTION_NAME]]
; CHECK-NEXT:    `define [[OPTION_NAME]]
; CHECK-NEXT:  `endif
; CHECK-NEXT:  `ifdef [[TARGET_NAME]]
; CHECK-NEXT:    `_ERROR_[[TARGET_NAME]]_must_not_be_set
; CHECK-NEXT:  `else 
; CHECK-NEXT:    `define [[TARGET_NAME]] proc_FPGA
; CHECK-NEXT:  `endif
