// RUN: circt-verilog %s | circt-opt -export-verilog -o /dev/null | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: (* flag_attr, str_attr = "hello \"world\"", int_attr = 42 *)
// CHECK-NEXT: module ModuleWithAttrs(
(* flag_attr, str_attr = "hello \"world\"", int_attr = 42 *)
module ModuleWithAttrs(input logic a, output logic b);
  assign b = a;
endmodule
