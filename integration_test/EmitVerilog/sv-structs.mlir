// REQUIRES: verilator
// RUN: circt-translate %s -export-verilog -verify-diagnostics > %t1.sv
// RUN: circt-rtl-sim.py %t1.sv --cycles 1 2>&1 | FileCheck %s

hw.module @top(%clk: i1, %rstn: i1) {
  %reg1 = sv.reg : !hw.inout<struct<a: i1, b: i1>>
  %reg2 = sv.reg : !hw.inout<array<2xstruct<a: i1, b: i1>>>

  %a = hw.constant 0 : i1
  %b = hw.constant 1 : i1
  %cst1 = hw.struct_create(%a, %b) : !hw.struct<a: i1, b: i1>
  %cst2 = hw.array_create %cst1, %cst1 : !hw.struct<a: i1, b: i1>

  sv.always posedge %clk {
    sv.passign %reg1, %cst1 : !hw.struct<a: i1, b: i1>
    sv.passign %reg2, %cst2 : !hw.array<2xstruct<a: i1, b: i1>>
    sv.fwrite "reg1: %b\n" (%reg1) : !hw.inout<struct<a: i1, b: i1>>
    sv.fwrite "reg2: %b\n" (%reg2) : !hw.inout<array<2xstruct<a: i1, b: i1>>>
  }

  // CHECK: [driver] Starting simulation
  // CHECK-NEXT: reg1: 00
  // CHECK-NEXT: reg2: 0000
  // CHECK-NEXT: reg1: 01
  // CHECK-NEXT: reg2: 0101
}
