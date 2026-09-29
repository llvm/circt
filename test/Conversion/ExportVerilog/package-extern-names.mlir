// RUN: split-file %s %t
// RUN: circt-opt %t/input.mlir -export-verilog -o %t/exported.mlir > %t/consumer.sv
// RUN: FileCheck %s --check-prefix=SV < %t/consumer.sv
// RUN: FileCheck %s --check-prefix=IR < %t/exported.mlir
// RUN: circt-opt %t/exported.mlir -export-verilog -o /dev/null | FileCheck %s --check-prefix=SV
// RUN: not circt-opt %t/invalid.mlir -export-verilog -o /dev/null > %t/invalid.sv 2> %t/invalid.err
// RUN: FileCheck %s --check-prefix=ERR < %t/invalid.err
// RUN: test ! -s %t/invalid.sv
// RUN: not circt-opt %t/invalid.mlir -export-split-verilog="dir-name=%t/invalid" -o /dev/null 2> %t/invalid-split.err
// RUN: FileCheck %s --check-prefix=ERR < %t/invalid-split.err
// RUN: test ! -e %t/invalid/InvalidConsumer.sv

// IR: hw.typedecl @word, "\\word-type" : i8
// IR: hw.typedecl @state, "\\State-Type" : !hw.enum<Idle, Busy>
// IR: hw.typedecl @mixed, "Module" : i8
// SV-NOT: {{^}}package
// SV-LABEL: module Consumer(
// SV: input {{ *}}\Ext-Types ::\word-type word
// SV: input {{ *}}\Ext-Types ::\word-type [1:0] words
// SV: input {{ *}}Normal::Module mixed
// SV: output \Ext-Types ::\State-Type state
// SV: output \Ext-Types ::plain plain
// SV: output Normal::\State-Type normal
// SV: assign state = \Ext-Types ::\State-Type_Idle ;
// SV: assign plain = \Ext-Types ::plain_Idle;
// SV: assign normal = Normal::\State-Type_Idle ;
// SV-NOT: {{^}}package
// ERR: external package member name "Module" is not allowed in Verilog output

//--- input.mlir
sv.package.extern @pkg {
  hw.typedecl @word, "\\word-type" : i8
  hw.typedecl @state, "\\State-Type" : !hw.enum<Idle, Busy>
  hw.typedecl @plain : !hw.enum<Idle, Busy>
} {hw.verilogName = "\\Ext-Types"}
sv.package.extern @Normal {
  hw.typedecl @state, "\\State-Type" : !hw.enum<Idle, Busy>
  hw.typedecl @mixed, "Module" : i8
}

hw.module @Consumer(
    in %word: !hw.typealias<@pkg::@word, i8>,
    in %words: !hw.array<2x!hw.typealias<@pkg::@word, i8>>,
    in %mixed: !hw.typealias<@Normal::@mixed, i8>,
    out state: !hw.typealias<@pkg::@state, !hw.enum<Idle, Busy>>,
    out plain: !hw.typealias<@pkg::@plain, !hw.enum<Idle, Busy>>,
    out normal: !hw.typealias<@Normal::@state, !hw.enum<Idle, Busy>>) {
  %state = hw.enum.constant Idle : !hw.typealias<@pkg::@state, !hw.enum<Idle, Busy>>
  %plain = hw.enum.constant Idle : !hw.typealias<@pkg::@plain, !hw.enum<Idle, Busy>>
  %normal = hw.enum.constant Idle : !hw.typealias<@Normal::@state, !hw.enum<Idle, Busy>>
  hw.output %state, %plain, %normal : !hw.typealias<@pkg::@state, !hw.enum<Idle, Busy>>, !hw.typealias<@pkg::@plain, !hw.enum<Idle, Busy>>, !hw.typealias<@Normal::@state, !hw.enum<Idle, Busy>>
}

//--- external.sv
package \Ext-Types ;
  typedef logic [7:0] \word-type ;
  typedef enum bit { \State-Type_Idle , \State-Type_Busy } \State-Type ;
  typedef enum bit { plain_Idle, plain_Busy } plain;
endpackage

package Normal;
  typedef enum bit { \State-Type_Idle , \State-Type_Busy } \State-Type ;
  typedef logic [7:0] Module;
endpackage

//--- invalid.mlir
module attributes {circt.loweringOptions = "caseInsensitiveKeywords"} {
  sv.package.extern @Types {
    hw.typedecl @word, "Module" : i8
  }
  hw.module @InvalidConsumer(in %word: !hw.typealias<@Types::@word, i8>) {}
}
