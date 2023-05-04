// RUN: circt-opt --export-verilog %s | FileCheck %s

// CHECK-LABEL: module symbols
// CHECK-NEXT: input baz /* inner_sym: bazSym */
module attributes {circt.loweringOptions="printDebugInfo"} {
hw.module @symbols(%baz: i1 {hw.exportPort = #hw<innerSym@bazSym>}) -> () {
    // CHECK: wire foo /* inner_sym: fooSym */;
    %foo = sv.wire sym @fooSym : !hw.inout<i1>
    // CHECK: reg bar /* inner_sym: barSym */;
    %bar = sv.reg sym @barSym : !hw.inout<i1>
}
}
