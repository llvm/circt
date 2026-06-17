// RUN: split-file %s %t
//
// A single unreferenced top module is unambiguous: --detect-top succeeds and
// the module lowers as usual.
// RUN: circt-verilog --detect-top %t/single.sv | FileCheck %s --check-prefix=SINGLE
//
// Multiple unreferenced top modules are ambiguous: --detect-top errors out.
// RUN: not circt-verilog --detect-top %t/multi.sv 2>&1 | FileCheck %s --check-prefix=MULTI
//
// The SAME multi-top input WITHOUT --detect-top still elaborates (the default
// path is unchanged).
// RUN: circt-verilog %t/multi.sv | FileCheck %s --check-prefix=DEFAULT
//
// REQUIRES: slang
// UNSUPPORTED: valgrind

//--- single.sv
// SINGLE: hw.module @Top
module Top(input logic a, output logic b);
  assign b = a;
endmodule

//--- multi.sv
// MULTI: expected exactly one runnable top module; found 2
// DEFAULT-DAG: hw.module @TopA
// DEFAULT-DAG: hw.module @TopB
module TopA(input logic a, output logic b);
  assign b = a;
endmodule
module TopB(input logic c, output logic d);
  assign d = c;
endmodule
