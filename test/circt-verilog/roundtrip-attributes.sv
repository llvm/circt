// RUN: circt-verilog %s | circt-opt -export-verilog -o /dev/null | FileCheck %s
// RUN: circt-verilog --record-specialization %s | circt-opt -export-verilog -o /dev/null | FileCheck %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// CHECK-LABEL: module AttrChild(
module AttrChild(input logic a, output logic b);
  assign b = a;
endmodule

// CHECK-LABEL: (* flag_attr, str_attr = "hello \"world\"", int_attr = 42 *)
// CHECK-NEXT: module ModuleWithAttrs(
(* flag_attr, str_attr = "hello \"world\"", int_attr = 42 *)
module ModuleWithAttrs #(parameter int W = 1) (
  input logic a,
  output logic b,
  output logic [1:0] c
);
  // CHECK: (* keep_hierarchy, mark = "val", weight = 7 *)
  // CHECK-NEXT: AttrChild u0 (
  (* keep_hierarchy, mark = "val", weight = 7 *)
  AttrChild u0(.a(a), .b(b));
  // CHECK: (* keep_hierarchy *)
  // CHECK-NEXT: AttrChild arr_0 (
  // CHECK: (* keep_hierarchy *)
  // CHECK-NEXT: AttrChild arr_1 (
  (* keep_hierarchy *)
  AttrChild arr [1:0] (.a(a), .b(c));
endmodule
