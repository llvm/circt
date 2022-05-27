// RUN: circt-opt --lowering-options=printDebugInfo --export-verilog %s | FileCheck %s

// CHECK-LABEL: module symbols
// CHECK-NEXT: input baz /* inner_sym: bazSym */
hw.module @symbols(%baz: i1 {hw.exportPort = @bazSym}) -> () {
    // CHECK: wire foo /* inner_sym: fooSym */;
    %foo = sv.wire sym @fooSym : !hw.inout<i1>
    // CHECK: reg bar /* inner_sym: barSym */;
    %bar = sv.reg sym @barSym : !hw.inout<i1>
}
