// RUN: circt-opt %s -export-verilog -o /dev/null | FileCheck %s --check-prefixes=CHECK,GROUP
// RUN: circt-opt %s -export-split-verilog="dir-name=%t" -o /dev/null
// RUN: FileCheck %s --check-prefix=LIST < %t/filelist.f
// RUN: FileCheck %s --check-prefix=GROUP < %t/grouped.sv

// Neither file discovery nor operation order gives packages priority.
// CHECK-LABEL: module Before(
hw.module @Before() {}
// CHECK-LABEL: package types;
sv.package @types {
  hw.typedecl @word : i8
}
// CHECK-LABEL: module Consumer(
// CHECK: input types::word word
hw.module @Consumer(in %word: !hw.typealias<@types::@word, i8>) {}

sv.package @grouped {
  hw.typedecl @word : !hw.int<#hw.param.verbatim<"`WIDTH">>
}
hw.module @Grouped(in %word: !hw.typealias<@grouped::@word, !hw.int<#hw.param.verbatim<"`WIDTH">>>) {}
// Explicit file contents, including required macro preambles, stay in order.
emit.file "grouped.sv" {
  emit.verbatim "`define WIDTH 8"
  emit.ref @grouped
  emit.ref @Grouped
} {output_file = #hw.output_file<"grouped.sv">}

// GROUP: `define WIDTH 8
// GROUP-NEXT: package grouped;
// GROUP-NEXT: typedef logic [`WIDTH - 1:0] word;
// GROUP-NEXT: endpackage
// GROUP-NEXT: module Grouped(
// GROUP: input grouped::word word
// GROUP-NOT: {{^}}package
// LIST: Before.sv
// LIST-NEXT: types.sv
// LIST-NEXT: Consumer.sv
// LIST-NEXT: grouped.sv
// LIST-NOT: .sv
