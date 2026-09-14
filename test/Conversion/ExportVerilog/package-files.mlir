// RUN: circt-opt %s -export-verilog -o /dev/null | FileCheck %s
// RUN: circt-opt %s -export-split-verilog="dir-name=%t" -o %t.mlir
// RUN: FileCheck %s --check-prefix=LIST < %t/filelist.f
// RUN: FileCheck %s --check-prefix=TYPES < %t/types.sv
// RUN: FileCheck %s --check-prefix=GROUPED < %t/grouped.sv
// RUN: FileCheck %s --check-prefix=FIRST < %t/First.sv
// RUN: circt-opt %t.mlir -export-split-verilog="dir-name=%t.again" -o /dev/null
// RUN: FileCheck %s --check-prefix=LIST < %t.again/filelist.f
// RUN: FileCheck %s --check-prefix=GROUPED < %t.again/grouped.sv

emit.fragment @Width {
  sv.verbatim "`define WIDTH 8"
}

// A replicated operation is not pulled into a file which only holds a package.
sv.verbatim "// Replicated into consumers."

// This file is collected before the package files but must follow them.
hw.module @First(in %b: !hw.typealias<@grouped::@byte, i8>) {}

// Packages precede the other contents of the file they belong to, and the
// files defining them come first in the generated file list.
// CHECK-LABEL: package types;
// CHECK-NEXT:    `WIDTH
// CHECK-NEXT:  endpackage
// CHECK-LABEL: module First(
// CHECK: grouped::byte_0 b
// TYPES-NOT:   Replicated into consumers.
// TYPES:       `define WIDTH 8
// TYPES:       package types;
// TYPES:       endpackage
// TYPES-NOT:   Replicated into consumers.
sv.package @types {
  hw.typedecl @word : !hw.int<#hw.param.verbatim<"`WIDTH">>
} {emit.fragments = [@Width]}

// A package explicitly assigned to a file is emitted there, once.
// GROUPED:      package grouped;
// GROUPED-NEXT:   typedef logic [7:0] byte_0;
// GROUPED-NEXT: endpackage
// GROUPED-NEXT: package second;
// GROUPED-NEXT:   typedef logic flag;
// GROUPED-NEXT: endpackage
// GROUPED-NEXT: module Grouped
// GROUPED: // After the consumer.
// GROUPED-NOT:  {{^}}package
sv.package @grouped {
  hw.typedecl @byte : i8
}
hw.module @Grouped(in %b: !hw.typealias<@grouped::@byte, i8>) {}
sv.package @second {
  hw.typedecl @flag : i1
}
emit.file "grouped.sv" {
  emit.ref @Grouped
  emit.ref @grouped
  emit.verbatim "// After the consumer."
  emit.ref @second
} {output_file = #hw.output_file<"grouped.sv">}

// CHECK-LABEL: module Consumer(
// CHECK:         input types::word
hw.module @Consumer(in %w: !hw.typealias<@types::@word, !hw.int<#hw.param.verbatim<"`WIDTH">>>) {}

// LIST:      types.sv
// LIST-NEXT: grouped.sv
// LIST-NEXT: First.sv
// LIST-NEXT: Consumer.sv
// LIST-NOT:  grouped.sv
// FIRST: module First(
// FIRST: grouped::byte_0 b
// FIRST-NOT: {{^}}package
