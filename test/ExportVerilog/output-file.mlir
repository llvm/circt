// RUN: circt-opt %s --export-verilog | FileCheck %s

// CHECK-LABEL: module B

#file1 = #hw.output_file<"dir1/file1.sv", includeReplicatedOps>
#filelist1 = #hw.output_filelist<"dir5/foo.f"> 
hw.module @foo1(%a: i1) -> (b: i1) attributes {output_file = #file1, output_filelist = #filelist1} {
  hw.output %a : i1
}
// CHECK-LABEL: FILE "dir1{{.}}file1.sv"
// CHECK-LABEL: module foo1
// CHECK-LABEL: module B

#file2 = #hw.output_file<"dir1/file2.sv">
#filelist2 = [#hw.output_filelist<"dir5/foo.f"> ,#hw.output_filelist<"dir5/bar.f">]
hw.module @foo2(%a: i1) -> (b: i1) attributes {output_file = #file2, output_filelist = #filelist2} {
  hw.output %a : i1
}
// CHECK-LABEL: FILE "dir1{{.}}file2.sv"
// CHECK-LABEL: module foo2

#file3 = #hw.output_file<"dir2/", includeReplicatedOps>
#filelist3 = [#hw.output_filelist<"dir5/bar.f">]
hw.module @foo3(%a: i1) -> (b: i1) attributes {output_file = #file3, output_filelist = #filelist3} {
  hw.output %a : i1
}
// CHECK-LABEL: FILE "dir2{{.}}foo3.sv"
// CHECK-LABEL: module foo3
// CHECK-LABEL: module B

#file4 = #hw.output_file<"file4.sv", includeReplicatedOps>
sv.verbatim "module A; endmodule" {output_file = #file4}
sv.verbatim "module B; endmodule"
// CHECK-LABEL: FILE "file4.sv"
// CHECK-LABEL: module A
// CHECK-LABEL: module B

// Absolute file names should override directories.
sv.verbatim "module C; endmodule" {output_file = #hw.output_file<"/tmp/dummy.sv">}
// CHECK-LABEL: FILE "/tmp/dummy.sv"
// CHECK-LABEL: module C

// Filelists
// CHECK-LABEL: FILE "dir5/foo.f"
// CHECK-EMPTY:
// CHECK-NEXT: dir1/file1.sv
// CHECK-NEXT: dir1/file2.sv

// CHECK-LABEL: FILE "dir5/bar.f"
// CHECK-EMPTY:
// CHECK-NEXT: dir1/file2.sv
// CHECK-NEXT: dir2/foo3.sv