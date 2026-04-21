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
    ; CHECK-NEXT:  `ifdef targets$Platform$FPGA
    ; CHECK-NEXT:    FPGATarget proc_FPGA (
    ; CHECK-NEXT:      .result_0 ([[RESULT]])
    ; CHECK-NEXT:    );
    ; CHECK-NEXT:    `define targets$Platform$top$proc proc_FPGA
    ; CHECK-NEXT:  `else  // targets$Platform$FPGA
    ; CHECK-NEXT:   $error("Required instance choice option 'Platform' not selected, must define one of: 'targets$Platform$FPGA'");
    ; CHECK-NEXT:  `endif // targets$Platform$FPGA
    instchoice proc of DefaultTarget, Platform:
      FPGA => FPGATarget

    printf(clk, UInt<1>(1), "result: %d\n", proc.result[0])
