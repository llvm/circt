; REQUIRES: verilator
; This test verifies that the instance choice macro-based mechanism works correctly:

; -------------------------------------- Setup
; RUN: rm -rf %t && mkdir -p %t
; RUN: firtool %s --split-verilog -o=%t

; -------------------------------------- Test 1: Multiple options error check
; RUN: not verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.multi.exe --top-module top -Dtargets\$Platform\$ASIC -Dtargets\$Platform\$FPGA 2>&1 | FileCheck %s --check-prefix=MULTI
; MULTI: Multiple instance choice options defined for option 'Platform': 'targets$Platform$ASIC' and 'targets$Platform$FPGA'

; -------------------------------------- Test 2: ASIC target
; RUN: verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.asic.exe --top-module top -Dtargets\$Platform\$ASIC
; RUN: %t.asic.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=ASIC
; ASIC: result:         10

; -------------------------------------- Test 3: FPGA target
; RUN: verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.fpga.exe --top-module top -Dtargets\$Platform\$FPGA
; RUN: %t.fpga.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=FPGA
; FPGA: result:         20

; -------------------------------------- Test 4: No option selected (error case)
; RUN: not verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.error.exe --top-module top 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: Required instance choice option 'Platform' not selected, must define one of: 'targets$Platform$ASIC', 'targets$Platform$FPGA'

FIRRTL version 5.1.0
circuit top:
  option Platform:
    FPGA
    ASIC

  module DefaultTarget:
    output result: UInt<32>

    connect result, UInt<32>(0)

  module ASICTarget:
    output result: UInt<32>

    connect result, UInt<32>(10)

  module FPGATarget:
    output result: UInt<32>

    connect result, UInt<32>(20)

  public module top:
    input clk: Clock
    input rst: UInt<1>

    instchoice proc of DefaultTarget, Platform:
      ASIC => ASICTarget
      FPGA => FPGATarget

    printf(clk, UInt<1>(1), "result: %d\n", proc.result)
