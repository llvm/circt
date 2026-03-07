; REQUIRES: verilator
; This test verifies that the instance choice header inclusion mechanism works correctly:

; -------------------------------------- Setup
; RUN: rm -rf %t && mkdir -p %t
; RUN: firtool %s --split-verilog -o=%t

; -------------------------------------- Test 1: Invalid includes
; RUN: not verilator %t/targets-top-Platform-ASIC.svh %t/targets-top-Platform-FPGA.svh %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --lint-only --top-module top 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: not verilator %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv %t/targets-top-Platform-ASIC.svh --cc --sv --exe --build -o %t.default.exe --top-module top 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: must_not_be_set

; -------------------------------------- Test 2: ASIC target
; RUN: verilator %driver %t/targets-top-Platform-ASIC.svh %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.asic.exe --top-module top
; RUN: %t.asic.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=ASIC
; ASIC: result:         10

; -------------------------------------- Test 3: FPGA target
; RUN: verilator %driver %t/targets-top-Platform-FPGA.svh %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.fpga.exe --top-module top
; RUN: %t.fpga.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=FPGA
; FPGA: result:         20

; -------------------------------------- Test 4: Default target
; RUN: verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.default.exe --top-module top
; RUN: %t.default.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=DEFAULT
; DEFAULT: result:       0

; -------------------------------------- Test 5: Configuration file
; RUN: echo '`include "targets-top-Platform-FPGA.svh"' > %t/user-config.svh
; RUN: verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --cc --sv --exe --build -o %t.config.exe --top-module top -DFIRRTL_CONFIGURATION_FILE=user-config.svh -I%t
; RUN: %t.config.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=CONFIGFILE
; CONFIGFILE: result:       20

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
