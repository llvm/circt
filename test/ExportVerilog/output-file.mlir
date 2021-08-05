// RUN: circt-translate %s --export-verilog --verify-diagnostics | FileCheck %s

// CHECK-LABEL: module B

// CHECK-LABEL: FILE "dir1{{.}}file1.sv"
// CHECK-LABEL: module foo1
// CHECK-LABEL: module B
// CHECK-LABEL: FILE "dir1{{.}}file2.sv"
// CHECK-LABEL: module foo2

// CHECK-LABEL: FILE "dir2{{.}}foo3.sv"
// CHECK-LABEL: module foo3
// CHECK-LABEL: module B

// CHECK-LABEL: FILE "file4.sv"
// CHECK-LABEL: module A
// CHECK-LABEL: module B

// CHECK-LABEL: FILE "/tmp/dummy.sv"
// CHECK-LABEL: module C

#file1 = {directory = "dir1", name = "file1.sv", exclude_replicated_ops = false}
#file2 = {directory = "dir1", name = "file2.sv"}
#file3 = {directory = "dir2", exclude_replicated_ops = false}
#file4 = {name = "file4.sv", exclude_replicated_ops = false}

hw.module @foo1(%a: i1) -> (%b: i1) attributes {output_file = #file1} {
  hw.output %a : i1
}
hw.module @foo2(%a: i1) -> (%b: i1) attributes {output_file = #file2} {
  hw.output %a : i1
}
hw.module @foo3(%a: i1) -> (%b: i1) attributes {output_file = #file3} {
  hw.output %a : i1
}

sv.verbatim "module A; endmodule" {output_file = #file4}
sv.verbatim "module B; endmodule"

// Absolute file names should override directories.
sv.verbatim "module C; endmodule" {output_file = {directory = "dir3", name = "/tmp/dummy.sv"}}
