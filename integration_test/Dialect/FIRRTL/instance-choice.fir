; REQUIRES: verilator
; This test verifies that the instance choice header inclusion mechanism works correctly
; with probes to access internal signals:

; RUN: rm -rf %t && mkdir -p %t
; RUN: firtool %s --split-verilog -o=%t
; Check it errors when including both headers.
; RUN: not verilator %t/targets_top_Platform_ASIC.svh %t/targets_top_Platform_FPGA.svh %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv --lint-only --top-module top 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: verilator %driver %t/targets_top_Platform_ASIC.svh %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv %t/ref_top.sv --cc --sv --exe --build -o %t.asic.exe --top-module top
; RUN: %t.asic.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=ASIC
; RUN: verilator %driver %t/targets_top_Platform_FPGA.svh %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv %t/ref_top.sv --cc --sv --exe --build -o %t.fpga.exe --top-module top
; RUN: %t.fpga.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=FPGA
; RUN: verilator %driver %t/top.sv %t/ASICTarget.sv %t/FPGATarget.sv %t/DefaultTarget.sv %t/ref_top.sv --cc --sv --exe --build -o %t.default.exe --top-module top
; RUN: %t.default.exe --cycles 1 2>&1 | FileCheck %s --check-prefix=DEFAULT
;
; ERROR: must__not__be__set
;
; ASIC: result:         10
; ASIC: internal:       15
; FPGA: result:         20
; FPGA: internal:       25
; DEFAULT: result:       0
; DEFAULT: internal:      5

FIRRTL version 5.1.0
circuit top:
  option Platform:
    FPGA
    ASIC

  module DefaultTarget:
    output result: UInt<32>
    output internal_probe: Probe<UInt<32>>

    wire internal: UInt<32>
    connect internal, UInt<32>(5)
    connect result, UInt<32>(0)
    define internal_probe = probe(internal)

  module ASICTarget:
    output result: UInt<32>
    output internal_probe: Probe<UInt<32>>

    wire internal: UInt<32>
    connect internal, UInt<32>(15)
    connect result, UInt<32>(10)
    define internal_probe = probe(internal)

  module FPGATarget:
    output result: UInt<32>
    output internal_probe: Probe<UInt<32>>

    wire internal: UInt<32>
    connect internal, UInt<32>(25)
    connect result, UInt<32>(20)
    define internal_probe = probe(internal)

  public module top:
    input clk: Clock
    input rst: UInt<1>

    instchoice proc of DefaultTarget, Platform:
      ASIC => ASICTarget
      FPGA => FPGATarget

    wire probed_value: UInt<32>
    connect probed_value, read(proc.internal_probe)

    printf(clk, UInt<1>(1), "result: %d\n", proc.result)
    printf(clk, UInt<1>(1), "internal: %d\n", probed_value)
