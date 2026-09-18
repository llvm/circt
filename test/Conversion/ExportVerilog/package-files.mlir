// RUN: circt-opt %s -export-verilog -o /dev/null | FileCheck %s
// RUN: circt-opt %s -export-split-verilog="dir-name=%t" -o %t.mlir
// RUN: FileCheck %s --check-prefix=LIST < %t/filelist.f
// RUN: FileCheck %s --check-prefix=TYPES < %t/types.sv
// RUN: FileCheck %s --check-prefix=GROUPED < %t/grouped.sv
// RUN: FileCheck %s --check-prefix=FIRST < %t/First.sv
// RUN: FileCheck %s --check-prefix=SHARED < %t/shared.sv
// RUN: circt-opt %t.mlir -export-split-verilog="dir-name=%t.again" -o /dev/null
// RUN: FileCheck %s --check-prefix=LIST < %t.again/filelist.f
// RUN: FileCheck %s --check-prefix=GROUPED < %t.again/grouped.sv
// RUN: FileCheck %s --check-prefix=TYPES < %t.again/types.sv
// RUN: FileCheck %s --check-prefix=SHARED < %t.again/shared.sv

emit.fragment @Width {
  sv.verbatim "`define WIDTH 8"
}

// Package files use the same replication defaults as other generated files.
sv.verbatim "// Replicated into consumers."

// A package encountered later stays after this module in their shared file.
hw.module @SharedFirst()
    attributes {output_file = #hw.output_file<"shared.sv", includeReplicatedOps>} {}

// Files and operations keep their discovery order, with no package priority.
// CHECK-LABEL: module First(
hw.module @First() {}

// CHECK-LABEL: package types;
// CHECK-NEXT:    `WIDTH
// CHECK-NEXT:  endpackage
// TYPES:       // Replicated into consumers.
// TYPES:       `define WIDTH 8
// TYPES:       package types;
// TYPES:       endpackage
// TYPES-NOT:   Replicated into consumers.
sv.package @types {
  hw.typedecl @word : !hw.int<#hw.param.verbatim<"`WIDTH">>
} {emit.fragments = [@Width]}

// A package explicitly assigned to a file is emitted there, once.
// Explicit output_file attributes can disable replication.
// The macro preamble and references in emit.file must remain in order.
// GROUPED-NOT: Replicated into consumers.
// GROUPED:      `define GROUP_WIDTH 1
// GROUPED:      package grouped;
// GROUPED-NEXT:   typedef logic [7:0] byte_0;
// GROUPED-NEXT: endpackage
// GROUPED-NEXT: module Grouped
// GROUPED: // After the consumer.
// GROUPED-NEXT: package second;
// GROUPED-NEXT:   typedef logic [`GROUP_WIDTH - 1:0] flag;
// GROUPED-NEXT: endpackage
// GROUPED-NOT:  {{^}}package
// GROUPED-NOT: Replicated into consumers.
sv.package @grouped {
  hw.typedecl @byte : i8
}
hw.module @Grouped(in %b: !hw.typealias<@grouped::@byte, i8>) {}
sv.package @second {
  hw.typedecl @flag : !hw.int<#hw.param.verbatim<"`GROUP_WIDTH">>
}
emit.file "grouped.sv" {
  emit.verbatim "`define GROUP_WIDTH 1"
  emit.ref @grouped
  emit.ref @Grouped
  emit.verbatim "// After the consumer."
  emit.ref @second
} {output_file = #hw.output_file<"grouped.sv">}

// CHECK-LABEL: module Consumer(
// CHECK:         input types::word
hw.module @Consumer(in %w: !hw.typealias<@types::@word, !hw.int<#hw.param.verbatim<"`WIDTH">>>) {}

sv.package @shared {
  hw.typedecl @word : i8
} {output_file = #hw.output_file<"shared.sv", includeReplicatedOps>}
hw.module @SharedLast(in %w: !hw.typealias<@shared::@word, i8>)
    attributes {output_file = #hw.output_file<"shared.sv", includeReplicatedOps>} {}
sv.package @sharedSecond {
  hw.typedecl @word : !hw.typealias<@shared::@word, i8>
} {output_file = #hw.output_file<"shared.sv", includeReplicatedOps>}

// SHARED: // Replicated into consumers.
// SHARED: module SharedFirst(
// SHARED: package shared;
// SHARED: endpackage
// SHARED-NEXT: module SharedLast(
// SHARED: shared::word w
// SHARED: package sharedSecond;
// SHARED: endpackage
// SHARED-NOT: {{^}}package

// LIST:      shared.sv
// LIST-NEXT: First.sv
// LIST-NEXT: types.sv
// LIST-NEXT: grouped.sv
// LIST-NEXT: Consumer.sv
// LIST-NOT:  grouped.sv
// FIRST: module First(
// FIRST-NOT: {{^}}package
