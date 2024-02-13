// RUN: split-file %s %t
// RUN: circt-verilog %t/a.sv %t/b.sv -E | FileCheck %s --check-prefixes=CHECK-MULTI-UNIT
// RUN: circt-verilog %t/a.sv %t/b.sv -E --single-unit | FileCheck %s --check-prefixes=CHECK-SINGLE-UNIT
// REQUIRES: slang

// CHECK-MULTI-UNIT: import hello::undefined;
// CHECK-SINGLE-UNIT: import hello::defined;

//--- a.sv

`define HELLO

//--- b.sv

`ifdef HELLO
import hello::defined;
`else
import hello::undefined;
`endif
