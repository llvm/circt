// RUN: circt-opt %s -export-verilog -o /dev/null | FileCheck %s
// RUN: circt-opt %s -export-split-verilog="dir-name=%t" -o /dev/null
// RUN: FileCheck %s --check-prefix=LIST < %t/filelist.f
// RUN: FileCheck %s --check-prefix=TYPES < %t/types.sv
// RUN: FileCheck %s --check-prefix=GROUPED < %t/grouped.sv

emit.fragment @Width {
  sv.verbatim "`define WIDTH 8"
}

// A replicated operation is not pulled into a file which only holds a package.
sv.verbatim "// Replicated into consumers."

// Packages precede the other contents of the file they belong to, and the
// files defining them come first in the generated file list.
// CHECK-LABEL: package types;
// CHECK-NEXT:    `WIDTH
// CHECK-NEXT:  endpackage
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
// GROUPED-NEXT: module Grouped
// GROUPED-NOT:  {{^}}package
sv.package @grouped {
  hw.typedecl @byte : i8
}
hw.module @Grouped(in %b: !hw.typealias<@grouped::@byte, i8>) {}
emit.file "grouped.sv" {
  emit.ref @grouped
  emit.ref @Grouped
}

// CHECK-LABEL: module Consumer(
// CHECK:         input types::word
hw.module @Consumer(in %w: !hw.typealias<@types::@word, !hw.int<#hw.param.verbatim<"`WIDTH">>>) {}

// LIST:      types.sv
// LIST-NEXT: Consumer.sv
// LIST-NOT:  grouped.sv
