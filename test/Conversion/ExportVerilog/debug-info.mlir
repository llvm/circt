// RUN: circt-opt --export-verilog %s | FileCheck %s

// CHECK-LABEL: module symbols
// CHECK-NEXT: input baz /* #hw<innerSym@bazSym> */
module attributes {circt.loweringOptions="printDebugInfo"} {
hw.module @symbols(in %baz: i1 {hw.exportPort = #hw<innerSym@bazSym>}) {
    // CHECK: wire foo /* #hw<innerSym@fooSym> */;
    %foo = sv.wire sym @fooSym : !sv.net<i1>
    // CHECK: logic bar /* #hw<innerSym@barSym> */;
    %bar = sv.var sym @barSym : !sv.var<i1>
}
}
