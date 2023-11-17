// REQUIRES: verilator
// RUN: circt-opt %s -lower-seq-to-sv -export-verilog -verify-diagnostics -o %t2.mlir > %t1.sv
// RUN: verilator --lint-only +1364-1995ext+v %t1.sv
// RUN: verilator --lint-only +1364-2001ext+v %t1.sv
// RUN: verilator --lint-only +1364-2005ext+v %t1.sv
// RUN: verilator --lint-only +1800-2005ext+sv %t1.sv
// RUN: verilator --lint-only +1800-2009ext+sv %t1.sv
// RUN: verilator --lint-only +1800-2012ext+sv %t1.sv
// RUN: verilator --lint-only +1800-2017ext+sv %t1.sv

hw.module @top(in %clk: !seq.clock, in %rst: i1) {
  %cst0 = hw.constant 0 : i32
  %cst1 = hw.constant 1 : i32

  %true = hw.constant 1 : i1

  %rC = seq.firreg %nextC clock %clk sym @regC reset sync %rst, %cst0 : i32
  %rB = seq.firreg %nextB clock %clk sym @regB reset sync %rst, %cst0 : i32
  %rA = seq.firreg %nextA clock %clk sym @regA reset sync %rst, %cst0 : i32

  %nextA = comb.add %rA, %cst1 : i32
  %nextB = comb.add %rB, %rA : i32
  %nextC = comb.add %rC, %rB : i32

  %clock = seq.from_clock %clk
  sv.alwaysff(posedge %clock) {
    %fd = hw.constant 0x80000002 : i32
    sv.fwrite %fd, "%d %d %d\n"(%rA, %rB, %rC) : i32, i32, i32
  }

  hw.output
}
