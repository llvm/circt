// RUN: circt-opt %s -export-verilog -o %t.mlir | FileCheck %s --check-prefix=SV --implicit-check-not="package ExtTypes;" --implicit-check-not="package FileTypes;"
// RUN: FileCheck %s --check-prefix=IR < %t.mlir
// RUN: circt-opt %t.mlir -export-verilog -o /dev/null | FileCheck %s --check-prefix=SV --implicit-check-not="package ExtTypes;" --implicit-check-not="package FileTypes;"
// RUN: circt-opt %s -export-split-verilog="dir-name=%t" -o /dev/null
// RUN: FileCheck %s --check-prefix=LIST < %t/filelist.f
// RUN: FileCheck %s --check-prefix=GROUP --implicit-check-not="{{^}}package " < %t/grouped.sv
// RUN: test ! -e %t/ExtTypes.sv
// RUN: test ! -e %t/FileTypes.sv
// RUN: test ! -e %t/external.sv

// Reserved names in this compilation must not rename external package members.
sv.reserve_names ["Req", "State_Idle"]

// External package names are reserved before renaming emitted declarations.
// SV-LABEL: package ExtTypes_0;
sv.package @local {} {hw.verilogName = "ExtTypes"}

// IR: sv.package @ExtTypes {
// IR: hw.typedecl @Req, "Req" : !hw.struct<addr: i32, data: i8>
// IR: hw.typedecl @State : !hw.enum<Idle, Busy>
// IR: } {extern}
sv.package @ExtTypes {
  hw.typedecl @Req, "Req" : !hw.struct<addr: i32, data: i8>
  hw.typedecl @State : !hw.enum<Idle, Busy>
} {extern}

// Even an explicit output_file does not create a file for an external package.
sv.package @FileTypes {
  hw.typedecl @word : i8
} {extern, output_file = #hw.output_file<"external.sv">}

hw.module.extern @ExtMod(in %req: !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>)

// SV-LABEL: module Consumer(
// SV: input {{ *}}ExtTypes::Req req
// SV: input {{ *}}FileTypes::word word
// SV: output ExtTypes::State state
// SV: ExtMod ext (
hw.module @Consumer(
    in %req: !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>,
    in %word: !hw.typealias<@FileTypes::@word, i8>,
    out state: !hw.typealias<@ExtTypes::@State, !hw.enum<Idle, Busy>>) {
  hw.instance "ext" @ExtMod(req: %req: !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>) -> ()
  // SV: assign state = ExtTypes::State_Idle;
  %idle = hw.enum.constant Idle : !hw.typealias<@ExtTypes::@State, !hw.enum<Idle, Busy>>
  hw.output %idle : !hw.typealias<@ExtTypes::@State, !hw.enum<Idle, Busy>>
}

// Explicit emit.ref operations also suppress the package body, and may refer to
// an external package from multiple files.
emit.file "grouped.sv" {
  emit.ref @ExtTypes
  emit.ref @FileTypes
  emit.verbatim "// External packages are supplied by another tool."
} {output_file = #hw.output_file<"grouped.sv">}
emit.file "second.sv" {
  emit.ref @ExtTypes
} {output_file = #hw.output_file<"second.sv">}

// GROUP: // External packages are supplied by another tool.
// LIST: ExtTypes_0.sv
// LIST-NEXT: Consumer.sv
// LIST-NEXT: grouped.sv
// LIST-NEXT: second.sv
// LIST-NOT: .sv
