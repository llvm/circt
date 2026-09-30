// RUN: split-file %s %t
// RUN: circt-opt %t/package.mlir -export-verilog -o %t/package.exported.mlir > %t/package.sv
// RUN: sed 's/sv.package /sv.package.extern /' %t/package.mlir > %t/extern.mlir
// RUN: circt-opt %t/extern.mlir -export-verilog -o /dev/null > %t/extern.sv
// RUN: FileCheck %s --check-prefix=MEMBER < %t/extern.sv
// RUN: sed -n '/^module /,$p' %t/package.sv > %t/package-module.sv
// RUN: sed -n '/^module /,$p' %t/extern.sv > %t/extern-module.sv
// RUN: diff %t/package-module.sv %t/extern-module.sv
// RUN: sed 's/sv.package /sv.package.extern /' %t/package.exported.mlir > %t/extern-exported.mlir
// RUN: circt-opt %t/extern-exported.mlir -export-verilog -o /dev/null > %t/extern-exported.sv
// RUN: sed -n '/^module /,$p' %t/extern-exported.sv > %t/extern-exported-module.sv
// RUN: diff %t/package-module.sv %t/extern-exported-module.sv

// Converting either the original or already-legalized package to an external
// package must preserve the consumer's typedef and enum-member references.
// MEMBER-NOT: {{^}}package
// MEMBER-LABEL: module Consumer(
// MEMBER: input {{ *}}Types::Req_0 req
// MEMBER: input {{ *}}Types::_5Cword2Dtype word
// MEMBER: input {{ *}}Types::Module_0 mixed
// MEMBER: input {{ *}}Types::Word_0 duplicate
// MEMBER: output Types::State state
// MEMBER: assign state = Types::State_Idle_1;
// MEMBER-NOT: {{^}}package

//--- package.mlir
module attributes {circt.loweringOptions = "caseInsensitiveKeywords,locationInfoStyle=none"} {
  sv.reserve_names ["Req", "State_Idle"]
  sv.package @types {
    hw.typedecl @req, "Req" : i8
    hw.typedecl @word, "\\word-type" : i8
    hw.typedecl @mixed, "Module" : i8
    hw.typedecl @first, "Word" : i8
    hw.typedecl @second, "Word" : i8
    hw.typedecl @State : !hw.enum<Idle, Busy>
    hw.typedecl @State_Idle : i8
  } {hw.verilogName = "Types"}

  hw.module @Consumer(
      in %req: !hw.typealias<@types::@req, i8>,
      in %word: !hw.typealias<@types::@word, i8>,
      in %mixed: !hw.typealias<@types::@mixed, i8>,
      in %duplicate: !hw.typealias<@types::@second, i8>,
      out state: !hw.typealias<@types::@State, !hw.enum<Idle, Busy>>) {
    %state = hw.enum.constant Idle : !hw.typealias<@types::@State, !hw.enum<Idle, Busy>>
    hw.output %state : !hw.typealias<@types::@State, !hw.enum<Idle, Busy>>
  }
}
