// RUN: rm -rf %t
// RUN: firtool %s --format=mlir -split-verilog -o=%t
// RUN: FileCheck %s --check-prefix=VERILOG-FILE1 < %t/dir1/file1.sv
// RUN: FileCheck %s --check-prefix=VERILOG-FILE2 < %t/dir2/file2.sv
// RUN: FileCheck %s --check-prefix=VERILOG-FOO2 < %t/dir2/foo2.sv
// RUN: FileCheck %s --check-prefix=VERILOG-FOO3 < %t/foo3.sv
// RUN: FileCheck %s --check-prefix=LIST < %t/filelist.f

// LIST:      dir1/file1.sv
// LIST-NEXT: dir2/foo2.sv
// LIST-NEXT: foo3.sv
// LIST-NEXT: dir2/file2.sv

// VERILOG-FILE1-LABEL: module foo1
// VERILOG-FILE1-LABEL: module A

// VERILOG-FILE2-LABEL: module B
// VERILOG-FILE2-LABEL: package C

// VERILOG-FOO2-LABEL: module foo2
// VERILOG-FOO2-LABEL: module A

// VERILOG-FOO3-LABEL: module foo3
// VERILOG-FOO3-LABEL: module A

#dir1 = {path = "dir1"}
#dir2 = {path = "dir2/"}
#none_dir = {path = ""}

#file1 = {path = "file1.sv", exclude_replicated_ops = false}
#file2 = {path = "file2.sv"}

hw.module @foo1(%a: i1) -> (%b: i1) attributes {output_file = #file1, output_dir = #dir1} {
  hw.output %a : i1
}
hw.module @foo2(%a: i1) -> (%b: i1) attributes {output_dir = #dir2} {
  hw.output %a : i1
}
hw.module @foo3(%a: i1) -> (%b: i1) attributes {output_dir = #none_dir} {
  hw.output %a : i1
}

sv.verbatim "module A; endmodule" {}
sv.verbatim "module B; endmodule" {output_file = #file2, output_dir = #dir2}
sv.verbatim "package C; endpackage" {output_file = #file2, output_dir = #dir2}
