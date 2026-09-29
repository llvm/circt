// RUN: circt-opt %s -export-verilog -o %t.mlir | FileCheck %s --check-prefix=SV --implicit-check-not="package ExtTypes;" --implicit-check-not="package FileTypes;" --implicit-check-not="package Unused;"
// RUN: FileCheck %s --check-prefix=IR < %t.mlir
// RUN: circt-opt %t.mlir -export-verilog -o /dev/null | FileCheck %s --check-prefix=SV --implicit-check-not="package ExtTypes;" --implicit-check-not="package FileTypes;" --implicit-check-not="package Unused;"
// RUN: circt-opt %s -export-split-verilog="dir-name=%t" -o /dev/null
// RUN: FileCheck %s --check-prefix=LIST < %t/filelist.f
// RUN: FileCheck %s --check-prefix=CONSUMER < %t/Consumer.sv
// RUN: test ! -e %t/ExtTypes.sv
// RUN: test ! -e %t/Unused.sv
// RUN: test ! -e %t/external.sv

// Reserved names in this compilation must not rename external package members.
sv.reserve_names ["Req", "State_Idle"]

// External package names are reserved before renaming emitted declarations.
// SV-LABEL: package ExtTypes_0;
sv.package @local {} {hw.verilogName = "ExtTypes"}

// The private package's external name must be reserved, not its MLIR symbol.
// SV-LABEL: package FileTypes_0;
sv.package @local_file_types {} {hw.verilogName = "FileTypes"}

// IR: sv.package.extern @ExtTypes {
// IR: hw.typedecl @Req, "Req" : !hw.struct<addr: i32, data: i8>
// IR: hw.typedecl @State : !hw.enum<Idle, Busy>
// IR: }
sv.package.extern @ExtTypes {
  hw.typedecl @Req, "Req" : !hw.struct<addr: i32, data: i8>
  hw.typedecl @State : !hw.enum<Idle, Busy>
}

// An external package is not assigned a file, even with output_file set.
sv.package.extern @Unused {}
sv.package.extern @renamed_file_types {
  hw.typedecl @word : i8
} {hw.verilogName = "FileTypes", output_file = #hw.output_file<"external.sv">, sym_visibility = "private"}

// SV-LABEL: package LocalTypes;
// SV: typedef ExtTypes::Req Request;
sv.package @LocalTypes {
  hw.typedecl @Request : !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>
}

hw.module.extern @ExtMod(in %req: !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>)

// SV-LABEL: module Consumer(
// SV: input {{ *}}ExtTypes::Req req
// SV: input {{ *}}FileTypes::word word
// SV: output ExtTypes::State state
// SV: ExtMod ext (
// CONSUMER-LABEL: module Consumer(
// CONSUMER: input {{ *}}ExtTypes::Req req
// CONSUMER: input {{ *}}FileTypes::word word
// CONSUMER: output ExtTypes::State state
// CONSUMER: ExtMod ext (
hw.module @Consumer(
    in %req: !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>,
    in %word: !hw.typealias<@renamed_file_types::@word, i8>,
    out state: !hw.typealias<@ExtTypes::@State, !hw.enum<Idle, Busy>>) {
  hw.instance "ext" @ExtMod(req: %req: !hw.typealias<@ExtTypes::@Req, !hw.struct<addr: i32, data: i8>>) -> ()
  // SV: assign state = ExtTypes::State_Idle;
  // CONSUMER: assign state = ExtTypes::State_Idle;
  %idle = hw.enum.constant Idle : !hw.typealias<@ExtTypes::@State, !hw.enum<Idle, Busy>>
  hw.output %idle : !hw.typealias<@ExtTypes::@State, !hw.enum<Idle, Busy>>
}

// LIST: ExtTypes_0.sv
// LIST-NEXT: FileTypes_0.sv
// LIST-NEXT: LocalTypes.sv
// LIST-NEXT: Consumer.sv
// LIST-NOT: .sv
