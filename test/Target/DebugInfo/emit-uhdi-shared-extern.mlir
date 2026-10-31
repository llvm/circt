// RUN: circt-translate %s --emit-uhdi --uhdi-indent=0 | FileCheck %s

// Two external modules with the same Verilog name are one generated module,
// with a binding for each.

// CHECK:      "generated":{"BlackBox":{"kind":"extern"},"Top":{}}
// CHECK-SAME: "bindings":[{"original":"Ext1","generated":"BlackBox"
// CHECK-SAME: {"original":"Ext2","generated":"BlackBox"
hw.module.extern @Ext1(in %a: i1) attributes {verilogName = "BlackBox"}
hw.module.extern @Ext2(in %a: i1) attributes {verilogName = "BlackBox"}

hw.module @Top(in %a: i1) {
  hw.instance "e1" @Ext1(a: %a: i1) -> ()
  hw.instance "e2" @Ext2(a: %a: i1) -> ()
}
