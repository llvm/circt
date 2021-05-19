// RUN: circt-translate %s --export-verilog --verify-diagnostics | FileCheck %s

// CHECK-LABEL: module A;
// CHECK-LABEL: module D;
// CHECK-LABEL: FILE "file1.sv"
// CHECK-LABEL: module A;
// CHECK-LABEL: module B;
// CHECK-LABEL: module D;
// CHECK-LABEL: FILE "file2.sv"
// CHECK-LABEL: module C;
// CHECK-LABEL: package E;

#file1 = {path = "file1.sv", exclude_replicated_ops = false}
#file2 = {path = "file2.sv"}

sv.verbatim "module A; endmodule"
sv.verbatim "module B; endmodule" {output_file = #file1}
sv.verbatim "module C; endmodule" {output_file = #file2}
sv.verbatim "module D; endmodule"
sv.verbatim "package E; endpackage" {output_file = #file2}
